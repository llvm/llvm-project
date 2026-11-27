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
// per-iteration cleanup. No structured op is flattened.
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

/// A goto to a label in an enclosing region. It leaves `exitedRegions`,
/// innermost first.
struct Jump {
  cir::GotoOp gotoOp;
  Block *dest;
  SmallVector<Region *, 4> exitedRegions;
};

/// Routes gotos through a destination slot. Each op a goto leaves is followed
/// by a dispatch block that loads the slot and switches on it, so gotos to
/// different labels can share the same op.
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
  Block *getOrCreateExitBlock(Region &region);
  Dispatch &getOrCreateDispatchAfter(Operation *op);
  void addCase(Dispatch &dispatch, unsigned id,
               llvm::function_ref<void()> buildCase);
  void buildDispatch(Dispatch &dispatch, Location loc);
  void resetInCatchHandlers();

  cir::FuncOp func;
  IRRewriter &rewriter;
  cir::IntType s32Type;
  cir::AllocaOp slot;
  llvm::MapVector<Operation *, Dispatch> dispatchesAfter;
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

  for (Jump &jump : jumps)
    for (Region *region : jump.exitedRegions)
      getOrCreateDispatchAfter(getRegionOwner(*region));

  for (Jump &jump : jumps) {
    unsigned id = ids.lookup(jump.dest);
    Location loc = jump.gotoOp.getLoc();
    ArrayRef<Region *> exited = jump.exitedRegions;

    // Takes the goto on from the closest region holding both it and the
    // label: clear the slot and branch to the label.
    auto continueInward = [&] {
      storeId(loc, 0);
      cir::BrOp::create(rewriter, loc, jump.dest);
    };

    // After each op the goto leaves, either leave the next region out or go
    // on to the label.
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

    rewriter.setInsertionPoint(jump.gotoOp);
    storeId(loc, id);
    createExit(*exited.front(), loc);
    rewriter.eraseOp(jump.gotoOp);
  }

  for (auto &[op, dispatch] : dispatchesAfter)
    buildDispatch(dispatch, op->getLoc());

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

/// Returns the regions a goto leaves on its way out to the region of its label
/// `dest`, or std::nullopt if one of them cannot be left yet or the label is
/// in a region the goto would have to enter.
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

  Jump jump{gotoOp, dest, {}};
  Region *region = gotoOp->getParentRegion();
  auto *common = llvm::find(labelRegions, region);
  while (common == labelRegions.end()) {
    if (!region || !getRegionExit(*region))
      return std::nullopt;
    jump.exitedRegions.push_back(region);
    region = getRegionOwner(*region)->getParentRegion();
    common = llvm::find(labelRegions, region);
  }

  if (common != labelRegions.begin())
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
      ++numOutwardGotos;
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
