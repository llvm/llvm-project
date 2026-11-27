//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that replaces symbolic `cir.goto` operations
// with control-flow edges, so that the CFG and the region-branch interfaces
// describe every way into a block.
//
// A goto whose label is in the same region becomes a `cir.br`. Any other goto
// stores its label's id in a destination slot, leaves each region between it
// and the closest region that also holds the label through that region's
// normal exit, and is routed onward after each region's op by a dispatch on
// the slot:
//
//   cir.if %c {                       cir.store %zero, %slot
//     cir.goto "out"          ==>     cir.if %c {
//   }                                   cir.store %id, %slot
//   ...                                 cir.yield
//   ^bb1:                             }
//     cir.label "out"                 %v = cir.load %slot
//                                     cir.switch.flat %v : ^cont [
//                                       id: ^to_out ]
//
// Leaving a region the normal way runs exactly what a goto out of it runs:
// a cleanup scope's normal cleanup, a catch handler's end_catch, a loop's
// per-iteration cleanup.
//
// From there the goto enters each region between it and the label through
// that region's entry. The op is steered into the right region without
// evaluating anything the goto skips: its condition or switch value becomes an
// argument of a block right before it, which the goto path sets directly, and
// a while or for loop yields true from its condition without computing it.
// At the start of the region, after its allocas, a dispatch on the slot sends
// the goto on toward the label. No structured op is flattened.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/PatternMatch.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/CIRTransformUtils.h"
#include "clang/CIR/Interfaces/CIRLoopOpInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_GOTOELIMINATION
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

/// How control leaves a region of a structured op when a goto crosses it.
enum class RegionExit {
  /// `cir.break`: loop bodies, switch bodies and case regions.
  Break,
  /// `cir.yield` in place: cleanup scope bodies, whose cleanup collects
  /// yields from any block.
  Yield,
  /// `cir.br` to a block that holds only the region's final `cir.yield`:
  /// if, scope, try and catch handler regions, which must exit from their
  /// last block.
  ExitBlock,
};

} // namespace

/// Returns how to leave `region` on the way out of it, or std::nullopt if a
/// goto cannot leave it yet.
static std::optional<RegionExit> getRegionExit(Region &region) {
  Operation *op = region.getParentOp();
  if (isa<cir::SwitchOp>(op))
    return RegionExit::Break;
  // A cir.break in a case region binds to the switch only if the case is a
  // direct child of it.
  if (auto caseOp = dyn_cast<cir::CaseOp>(op))
    return isa<cir::SwitchOp>(caseOp->getParentOp())
               ? std::optional(RegionExit::Break)
               : std::nullopt;
  if (auto loop = dyn_cast<cir::LoopOpInterface>(op))
    return &region == &loop.getBody() ? std::optional(RegionExit::Break)
                                      : std::nullopt;
  if (auto cleanup = dyn_cast<cir::CleanupScopeOp>(op))
    return &region == &cleanup.getBodyRegion()
               ? std::optional(RegionExit::Yield)
               : std::nullopt;
  if (auto scope = dyn_cast<cir::ScopeOp>(op))
    return scope->getNumResults() == 0 ? std::optional(RegionExit::ExitBlock)
                                       : std::nullopt;
  if (isa<cir::IfOp, cir::TryOp>(op))
    return RegionExit::ExitBlock;
  return std::nullopt;
}

/// The op that holds `region` one level up: the switch for a case region,
/// whose body is not a level of its own, otherwise the region's parent.
static Operation *getRegionOwner(Region &region) {
  Operation *op = region.getParentOp();
  if (isa<cir::CaseOp>(op))
    return op->getParentOp();
  return op;
}

/// Returns the smallest value that no direct case of `switchOp` matches, or
/// std::nullopt if the cases cover every value of the condition's type.
static std::optional<APInt> getUnusedCaseValue(cir::SwitchOp switchOp) {
  auto condType = cast<cir::IntType>(switchOp.getCondition().getType());
  SmallVector<std::pair<APInt, APInt>> taken;
  for (Operation &op : switchOp.getBody().getOps()) {
    auto other = dyn_cast<cir::CaseOp>(op);
    if (!other)
      continue;
    ArrayAttr values = other.getValue();
    if (other.getKind() == cir::CaseOpKind::Range) {
      taken.emplace_back(cast<cir::IntAttr>(values[0]).getValue(),
                         cast<cir::IntAttr>(values[1]).getValue());
      continue;
    }
    for (Attribute value : values) {
      APInt v = cast<cir::IntAttr>(value).getValue();
      taken.emplace_back(v, v);
    }
  }
  APInt candidate(condType.getWidth(), 0);
  for (bool moved = true; moved;) {
    moved = false;
    for (auto &[lo, hi] : taken) {
      bool inside = condType.isSigned()
                        ? candidate.sge(lo) && candidate.sle(hi)
                        : candidate.uge(lo) && candidate.ule(hi);
      if (!inside)
        continue;
      if (hi.isAllOnes() || (condType.isSigned() && hi.isMaxSignedValue()))
        return std::nullopt;
      candidate = hi + 1;
      moved = true;
    }
  }
  return candidate;
}

/// Whether `caseOp` is a GNU range whose upper bound is below its lower one,
/// which no value matches.
static bool isEmptyRange(cir::SwitchOp switchOp, cir::CaseOp caseOp) {
  if (caseOp.getKind() != cir::CaseOpKind::Range)
    return false;
  APInt lo = cast<cir::IntAttr>(caseOp.getValue()[0]).getValue();
  APInt hi = cast<cir::IntAttr>(caseOp.getValue()[1]).getValue();
  return cast<cir::IntType>(switchOp.getCondition().getType()).isSigned()
             ? lo.sgt(hi)
             : lo.ugt(hi);
}

/// Returns the case value that steers `switchOp` into `caseOp`, or
/// std::nullopt if there is none: for the default case, a value no other case
/// matches; for an empty range, none.
static std::optional<APInt> getSteeringValue(cir::SwitchOp switchOp,
                                             cir::CaseOp caseOp) {
  if (isEmptyRange(switchOp, caseOp))
    return std::nullopt;
  if (caseOp.getKind() != cir::CaseOpKind::Default)
    return cast<cir::IntAttr>(caseOp.getValue()[0]).getValue();
  return getUnusedCaseValue(switchOp);
}

/// Whether a goto can enter `region` from the op that holds it.
static bool canEnter(Region &region) {
  Operation *op = region.getParentOp();
  if (auto scope = dyn_cast<cir::ScopeOp>(op))
    return scope->getNumResults() == 0;
  if (isa<cir::IfOp>(op))
    return true;
  if (auto loop = dyn_cast<cir::LoopOpInterface>(op))
    return &region == &loop.getBody() && !loop.maybeGetCleanup();
  if (auto cleanup = dyn_cast<cir::CleanupScopeOp>(op))
    return &region == &cleanup.getBodyRegion();
  if (auto caseOp = dyn_cast<cir::CaseOp>(op)) {
    auto switchOp = dyn_cast<cir::SwitchOp>(caseOp->getParentOp());
    return switchOp && getSteeringValue(switchOp, caseOp);
  }
  return false;
}

/// The operand of `op` that a goto steers, if any.
static OpOperand *getSteeredOperand(Operation *op) {
  if (auto ifOp = dyn_cast<cir::IfOp>(op))
    return &ifOp.getConditionMutable();
  if (auto switchOp = dyn_cast<cir::SwitchOp>(op))
    return &switchOp.getConditionMutable();
  return nullptr;
}

/// Whether control may reach `op` through a new block right before it
/// without breaking dominance. That holds unless an op at or after `op` uses
/// a value, other than an alloca or `op`'s own steered operand, that is
/// defined in `op`'s region before `op` or in another block of the region.
static bool canBranchTo(Operation *op) {
  Region *region = op->getParentRegion();
  Block *block = op->getBlock();
  OpOperand *steered = getSteeredOperand(op);
  auto isSafe = [&](Operation *user, Value value) {
    if (user == op && steered && value == steered->get())
      return true;
    Operation *def = value.getDefiningOp();
    Block *defBlock =
        def ? def->getBlock() : cast<BlockArgument>(value).getOwner();
    if (defBlock->getParent() != region || isa_and_present<cir::AllocaOp>(def))
      return true;
    return defBlock == block && def && !def->isBeforeInBlock(op);
  };
  for (Operation *at = op; at; at = at->getNextNode()) {
    WalkResult result = at->walk([&](Operation *user) {
      for (Value value : user->getOperands())
        if (!isSafe(user, value))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return false;
  }
  return true;
}

/// Makes `region` leave only from a final block that does nothing but yield,
/// as if, scope and try regions must, and returns that block. The region's
/// yields, which a case region may have in any block, branch to it.
static Block *sinkYieldsToLastBlock(Region &region, RewriterBase &rewriter) {
  // Blocks of a dispatch under construction have no terminator yet.
  SmallVector<cir::YieldOp> yields;
  for (Block &block : region)
    if (auto yield = dyn_cast_if_present<cir::YieldOp>(
            block.empty() ? nullptr : &block.back()))
      yields.push_back(yield);
  Block *last = &region.back();
  if (yields.size() == 1 && llvm::hasSingleElement(*last) &&
      yields.front()->getBlock() == last)
    return last;
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = region.getParentOp()->getLoc();
  Block *exit = rewriter.createBlock(&region, region.end());
  cir::YieldOp::create(rewriter, loc);
  for (cir::YieldOp yield : yields) {
    rewriter.setInsertionPoint(yield);
    rewriter.replaceOpWithNewOp<cir::BrOp>(yield, exit);
  }
  return exit;
}

namespace {

/// A goto to a label in another region. It leaves `exitedRegions`, innermost
/// first, then enters `enteredRegions`, outermost first.
struct Jump {
  cir::GotoOp gotoOp;
  Block *dest;
  SmallVector<Region *, 4> exitedRegions;
  SmallVector<Region *, 4> enteredRegions;
};

/// Routes gotos through a destination slot. Each op a goto leaves is followed
/// by a dispatch block that loads the slot and switches on it, and each region
/// a goto enters starts with one, so gotos to different labels can share the
/// same op.
class GotoRewriter {
public:
  GotoRewriter(cir::FuncOp func, IRRewriter &rewriter)
      : func(func), rewriter(rewriter),
        s32Type(cir::IntType::get(func.getContext(), 32, /*isSigned=*/true)) {}

  void rewrite(MutableArrayRef<Jump> jumps);

private:
  /// A dispatch on the slot: `cont` is where normal execution continues, and
  /// each destination id maps to the block that takes it on.
  struct Dispatch {
    Block *dispatchBlock;
    Block *cont;
    llvm::MapVector<unsigned, Block *> cases;
  };

  Value createConstant(Location loc, unsigned value);
  void storeId(Location loc, unsigned id);
  void createExit(Region &region, Location loc);
  void createEntry(Region &region, Location loc);
  Block *getOrCreateExitBlock(Region &region);
  Dispatch &getOrCreateDispatchAfter(Operation *op);
  Dispatch &getOrCreateDispatchAtEntry(Region &region);
  Block *getOrCreatePreBlock(Operation *op);
  Block *getOrCreateConditionBlock(Region &cond);
  void addCase(Dispatch &dispatch, unsigned id,
               llvm::function_ref<void()> buildCase);
  void buildDispatch(Dispatch &dispatch, Location loc);
  void resetInCatchHandlers();

  cir::FuncOp func;
  IRRewriter &rewriter;
  cir::IntType s32Type;
  cir::AllocaOp slot;
  llvm::MapVector<Operation *, Dispatch> dispatchesAfter;
  llvm::MapVector<Region *, Dispatch> dispatchesAtEntry;
  llvm::DenseMap<Operation *, Block *> preBlocks;
  llvm::DenseMap<Region *, Block *> conditionBlocks;
  llvm::DenseMap<Region *, Block *> exitBlocks;
};

} // namespace

Value GotoRewriter::createConstant(Location loc, unsigned value) {
  return cir::ConstantOp::create(rewriter, loc,
                                 cir::IntAttr::get(s32Type, value));
}

void GotoRewriter::storeId(Location loc, unsigned id) {
  cir::StoreOp::create(rewriter, loc, createConstant(loc, id), slot);
}

Block *GotoRewriter::getOrCreateExitBlock(Region &region) {
  Block *&exit = exitBlocks[&region];
  if (!exit)
    exit = sinkYieldsToLastBlock(region, rewriter);
  return exit;
}

void GotoRewriter::createExit(Region &region, Location loc) {
  switch (*getRegionExit(region)) {
  case RegionExit::Break:
    cir::BreakOp::create(rewriter, loc);
    return;
  case RegionExit::Yield:
    cir::YieldOp::create(rewriter, loc);
    return;
  case RegionExit::ExitBlock:
    cir::BrOp::create(rewriter, loc, getOrCreateExitBlock(region));
    return;
  }
}

void GotoRewriter::createEntry(Region &region, Location loc) {
  // Branch to the block right before the op that holds `region`, passing
  // the value that steers the op into `region`.
  Operation *owner = getRegionOwner(region);
  SmallVector<Value, 1> steering;
  if (auto ifOp = dyn_cast<cir::IfOp>(owner)) {
    bool toThen = &region == &ifOp.getThenRegion();
    steering.push_back(cir::ConstantOp::create(
        rewriter, loc, cir::BoolAttr::get(func.getContext(), toThen)));
  } else if (auto switchOp = dyn_cast<cir::SwitchOp>(owner)) {
    auto caseOp = cast<cir::CaseOp>(region.getParentOp());
    APInt value = *getSteeringValue(switchOp, caseOp);
    steering.push_back(cir::ConstantOp::create(
        rewriter, loc,
        cir::IntAttr::get(switchOp.getCondition().getType(), value)));
  }
  cir::BrOp::create(rewriter, loc, getOrCreatePreBlock(owner), steering);
}

GotoRewriter::Dispatch &GotoRewriter::getOrCreateDispatchAfter(Operation *op) {
  auto [it, inserted] = dispatchesAfter.try_emplace(op);
  Dispatch &dispatch = it->second;
  if (!inserted)
    return dispatch;

  // Clear the slot on every way into the op, so that the dispatch after it
  // never sees an id from an earlier goto that an exception interrupted.
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = op->getLoc();
  rewriter.setInsertionPoint(op);
  storeId(loc, 0);

  Block *block = op->getBlock();
  dispatch.cont = rewriter.splitBlock(block, std::next(op->getIterator()));
  dispatch.dispatchBlock = rewriter.createBlock(dispatch.cont);
  rewriter.setInsertionPointToEnd(block);
  cir::BrOp::create(rewriter, loc, dispatch.dispatchBlock);
  return dispatch;
}

GotoRewriter::Dispatch &
GotoRewriter::getOrCreateDispatchAtEntry(Region &region) {
  auto [it, inserted] = dispatchesAtEntry.try_emplace(&region);
  Dispatch &dispatch = it->second;
  if (!inserted)
    return dispatch;

  // The dispatch goes after the entry block's allocas, so they dominate
  // everything a goto reaches in the region.
  Block *entry = &region.front();
  auto firstNonAlloca = llvm::find_if(
      *entry, [](Operation &op) { return !isa<cir::AllocaOp>(op); });
  dispatch.cont = rewriter.splitBlock(entry, firstNonAlloca);
  dispatch.dispatchBlock = entry;
  return dispatch;
}

Block *GotoRewriter::getOrCreatePreBlock(Operation *op) {
  Block *&pre = preBlocks[op];
  if (pre)
    return pre;

  // Normal execution clears the slot and passes the op's own steered operand
  // on; a goto branches here with the slot set and a constant instead.
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = op->getLoc();
  Block *block = op->getBlock();
  pre = rewriter.splitBlock(block, op->getIterator());
  SmallVector<Value, 1> normalArgs;
  if (OpOperand *steered = getSteeredOperand(op)) {
    BlockArgument arg = pre->addArgument(steered->get().getType(), loc);
    normalArgs.push_back(steered->get());
    rewriter.modifyOpInPlace(op, [&] { steered->set(arg); });
  }
  rewriter.setInsertionPointToEnd(block);
  storeId(loc, 0);
  cir::BrOp::create(rewriter, loc, pre, normalArgs);
  return pre;
}

Block *GotoRewriter::getOrCreateConditionBlock(Region &cond) {
  Block *&conditionBlock = conditionBlocks[&cond];
  if (conditionBlock)
    return conditionBlock;

  // FlattenCFG expects one cir.condition per loop. Move it into a last block
  // that takes the condition as an argument, so a goto into the body can pass
  // true without evaluating the condition.
  auto condition = cast<cir::ConditionOp>(cond.back().getTerminator());
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = condition.getLoc();
  Value value = condition.getCondition();
  conditionBlock =
      rewriter.createBlock(&cond, cond.end(), value.getType(), loc);
  cir::ConditionOp::create(rewriter, loc, conditionBlock->getArgument(0));
  rewriter.setInsertionPoint(condition);
  rewriter.replaceOpWithNewOp<cir::BrOp>(condition, conditionBlock, value);
  return conditionBlock;
}

void GotoRewriter::addCase(Dispatch &dispatch, unsigned id,
                           llvm::function_ref<void()> buildCase) {
  Block *&caseBlock = dispatch.cases[id];
  if (caseBlock)
    return;
  OpBuilder::InsertionGuard guard(rewriter);
  caseBlock = rewriter.createBlock(dispatch.cont);
  buildCase();
}

void GotoRewriter::buildDispatch(Dispatch &dispatch, Location loc) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(dispatch.dispatchBlock);
  Value id = cir::LoadOp::create(rewriter, loc, slot);
  SmallVector<APInt> caseValues;
  SmallVector<Block *> caseDests;
  SmallVector<ValueRange> caseOperands;
  for (auto [caseId, caseBlock] : dispatch.cases) {
    caseValues.push_back(APInt(32, caseId, /*isSigned=*/true));
    caseDests.push_back(caseBlock);
    caseOperands.push_back(ValueRange());
  }
  cir::SwitchFlatOp::create(rewriter, loc, id, dispatch.cont, ValueRange(),
                            caseValues, caseDests, caseOperands);
}

void GotoRewriter::resetInCatchHandlers() {
  // An exception thrown while a goto is on its way out (by a destructor its
  // cleanup runs) can be caught in this function. The handler's normal
  // completion then continues past ops whose dispatch would otherwise see the
  // interrupted goto's id.
  func.walk([&](cir::BeginCatchOp beginCatch) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(beginCatch);
    storeId(beginCatch.getLoc(), 0);
  });
}

void GotoRewriter::rewrite(MutableArrayRef<Jump> jumps) {
  if (jumps.empty())
    return;

  // One i32 slot for the function, cleared after the entry block's allocas.
  // Ids are 1-based in label order; 0 means no goto is in flight.
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Location loc = func.getLoc();
    slot = cir::createDestSlot(func, rewriter, loc, "__goto_dest_slot");
    slot.setGotoDestSlot(true);
    rewriter.restoreInsertionPoint(
        cir::CIRBaseBuilderTy::getBestAllocaInsertPoint(
            &func.getBody().front()));
    storeId(loc, 0);
  }

  llvm::DenseMap<Block *, unsigned> ids;
  for (const Jump &jump : jumps)
    ids.try_emplace(jump.dest, ids.size() + 1);

  // Build the structure first: the clearing store before an op a goto leaves
  // must end up before the block a goto enters through, never inside it.
  for (Jump &jump : jumps)
    for (Region *region : jump.exitedRegions)
      getOrCreateDispatchAfter(getRegionOwner(*region));
  for (Jump &jump : jumps) {
    for (Region *region : jump.enteredRegions) {
      getOrCreatePreBlock(getRegionOwner(*region));
      getOrCreateDispatchAtEntry(*region);
      if (auto loop = dyn_cast<cir::LoopOpInterface>(region->getParentOp());
          loop && !isa<cir::DoWhileOp>(loop))
        getOrCreateDispatchAtEntry(loop.getCond());
    }
  }

  for (Jump &jump : jumps) {
    unsigned id = ids.lookup(jump.dest);
    Location loc = jump.gotoOp.getLoc();
    ArrayRef<Region *> exited = jump.exitedRegions;
    ArrayRef<Region *> entered = jump.enteredRegions;

    // Takes the goto on from the closest region holding both it and the
    // label: into the first entered region, or straight to the label.
    auto continueInward = [&] {
      if (entered.empty()) {
        storeId(loc, 0);
        cir::BrOp::create(rewriter, loc, jump.dest);
      } else {
        createEntry(*entered.front(), loc);
      }
    };

    // After each op the goto leaves, either leave the next region out or go
    // on inward.
    for (size_t level = 0, e = exited.size(); level != e; ++level) {
      Dispatch &dispatch =
          getOrCreateDispatchAfter(getRegionOwner(*exited[level]));
      addCase(dispatch, id, [&] {
        if (level + 1 == e)
          continueInward();
        else
          createExit(*exited[level + 1], loc);
      });
    }

    // At the start of each region the goto enters, either enter the next one
    // or, in the label's region, clear the slot and branch to the label. A
    // while or for loop first yields true from its condition.
    for (size_t level = 0, e = entered.size(); level != e; ++level) {
      Region &region = *entered[level];
      if (auto loop = dyn_cast<cir::LoopOpInterface>(region.getParentOp());
          loop && !isa<cir::DoWhileOp>(loop)) {
        addCase(getOrCreateDispatchAtEntry(loop.getCond()), id, [&] {
          Value enter = cir::ConstantOp::create(
              rewriter, loc, cir::BoolAttr::get(func.getContext(), true));
          cir::BrOp::create(rewriter, loc,
                            getOrCreateConditionBlock(loop.getCond()), enter);
        });
      }
      addCase(getOrCreateDispatchAtEntry(region), id, [&] {
        if (level + 1 == e) {
          storeId(loc, 0);
          cir::BrOp::create(rewriter, loc, jump.dest);
        } else {
          createEntry(*entered[level + 1], loc);
        }
      });
    }

    rewriter.setInsertionPoint(jump.gotoOp);
    storeId(loc, id);
    if (exited.empty())
      continueInward();
    else
      createExit(*exited.front(), loc);
    rewriter.eraseOp(jump.gotoOp);
  }

  for (auto &[op, dispatch] : dispatchesAfter)
    buildDispatch(dispatch, op->getLoc());
  for (auto &[region, dispatch] : dispatchesAtEntry)
    buildDispatch(dispatch, region->getParentOp()->getLoc());

  resetInCatchHandlers();
}

namespace {

struct GotoEliminationPass
    : public impl::GotoEliminationBase<GotoEliminationPass> {
  using GotoEliminationBase::GotoEliminationBase;

  void runOnOperation() override;
};

} // namespace

/// Erases the labels of `func` that no goto names and whose address is not
/// taken. Returns the number of labels erased.
static unsigned eraseUnusedLabels(cir::FuncOp func, RewriterBase &rewriter) {
  llvm::DenseSet<StringAttr> named;
  SmallVector<cir::LabelOp> labels;
  func.getBody().walk([&](Operation *op) {
    if (auto label = dyn_cast<cir::LabelOp>(op))
      labels.push_back(label);
    else if (auto gotoOp = dyn_cast<cir::GotoOp>(op))
      named.insert(gotoOp.getLabelAttr());
  });
  unsigned numErased = 0;
  for (cir::LabelOp label : labels) {
    if (label.getAddressTaken() || named.contains(label.getLabelAttr()))
      continue;
    rewriter.eraseOp(label);
    ++numErased;
  }
  return numErased;
}

/// Whether `op` is nested in an op of another dialect, such as an OpenMP or
/// OpenACC construct. The function's slot can't be shared with such regions:
/// some are isolated from above, others run on several threads.
static bool isInForeignOp(Operation *op) {
  for (Operation *parent = op->getParentOp(); !isa<cir::FuncOp>(parent);
       parent = parent->getParentOp())
    if (!isa<cir::CIRDialect>(parent->getDialect()))
      return true;
  return false;
}

/// Returns the regions a goto from `from` to a label in `to` leaves and
/// enters, or std::nullopt if one of them cannot be left or entered yet.
static std::optional<Jump> planJump(cir::GotoOp gotoOp, Block *dest) {
  if (isInForeignOp(gotoOp) || isInForeignOp(&dest->front()))
    return std::nullopt;

  // The label's regions, innermost first, up to the function body.
  SmallVector<Region *, 8> labelRegions;
  for (Region *region = dest->getParent(); region;
       region = getRegionOwner(*region)->getParentRegion()) {
    labelRegions.push_back(region);
    if (isa<cir::FuncOp>(region->getParentOp()))
      break;
  }

  Jump jump{gotoOp, dest, {}, {}};
  Region *region = gotoOp->getParentRegion();
  auto *common = llvm::find(labelRegions, region);
  while (common == labelRegions.end()) {
    if (!region || !getRegionExit(*region))
      return std::nullopt;
    jump.exitedRegions.push_back(region);
    region = getRegionOwner(*region)->getParentRegion();
    common = llvm::find(labelRegions, region);
  }

  for (auto it = std::make_reverse_iterator(common), e = labelRegions.rend();
       it != e; ++it) {
    if (!canEnter(**it) || !canBranchTo(getRegionOwner(**it)))
      return std::nullopt;
    jump.enteredRegions.push_back(*it);
  }
  if (!jump.enteredRegions.empty() && !canBranchTo(&dest->front()))
    return std::nullopt;
  return jump;
}

void GotoEliminationPass::runOnOperation() {
  cir::FuncOp func = getOperation();
  IRRewriter rewriter(func.getContext());

  llvm::DenseMap<StringAttr, cir::LabelOp> labels;
  SmallVector<cir::GotoOp> gotos;
  func.getBody().walk([&](Operation *op) {
    if (auto label = dyn_cast<cir::LabelOp>(op))
      labels.try_emplace(label.getLabelAttr(), label);
    else if (auto gotoOp = dyn_cast<cir::GotoOp>(op))
      gotos.push_back(gotoOp);
  });

  SmallVector<Jump> jumps;
  for (cir::GotoOp gotoOp : gotos) {
    Block *dest = labels.lookup(gotoOp.getLabelAttr())->getBlock();
    if (dest->getParent() == gotoOp->getParentRegion()) {
      rewriter.setInsertionPoint(gotoOp);
      rewriter.replaceOpWithNewOp<cir::BrOp>(gotoOp, dest);
      ++numSameRegionGotos;
      continue;
    }
    // A goto this cannot route yet is left for GotoSolver.
    if (std::optional<Jump> jump = planJump(gotoOp, dest)) {
      if (jump->enteredRegions.empty())
        ++numOutwardGotos;
      else
        ++numInwardGotos;
      jumps.push_back(std::move(*jump));
    }
  }

  GotoRewriter(func, rewriter).rewrite(jumps);

  if (!labels.empty())
    numErasedLabels += eraseUnusedLabels(func, rewriter);
}

std::unique_ptr<Pass> mlir::createGotoEliminationPass() {
  return std::make_unique<GotoEliminationPass>();
}
