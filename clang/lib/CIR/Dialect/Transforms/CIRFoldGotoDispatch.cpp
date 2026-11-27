//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that branches straight to the destination of a
// goto once the CFG is flat, by folding the dispatch on the goto destination
// slot that cir-goto-elimination introduced.
//
// The slot is only ever stored constants and loaded, so a forward dataflow
// over the function's blocks tells which constant it holds on each edge. An
// edge into a dispatch whose value is known goes straight to the block the
// dispatch would pick. That repeats until nothing changes, then dispatches
// nothing reaches and the slot, once nothing loads it, are erased.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/PatternMatch.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_CIRFOLDGOTODISPATCH
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

/// What is known about the slot's value at a program point.
class SlotValue {
public:
  /// No path into the point has been seen yet.
  SlotValue() = default;
  static SlotValue unvisited() { return SlotValue(); }
  static SlotValue overdefined() { return SlotValue(State::Overdefined); }
  static SlotValue constant(APInt value) {
    SlotValue result(State::Constant);
    result.value = std::move(value);
    return result;
  }

  bool isUnvisited() const { return state == State::Unvisited; }
  std::optional<APInt> getConstant() const {
    if (state != State::Constant)
      return std::nullopt;
    return value;
  }

  /// Joins the value of another path into the same point.
  SlotValue join(const SlotValue &other) const {
    if (isUnvisited())
      return other;
    if (other.isUnvisited() || *this == other)
      return *this;
    return overdefined();
  }

  bool operator==(const SlotValue &other) const {
    return state == other.state &&
           (state != State::Constant || value == other.value);
  }

private:
  enum class State { Unvisited, Constant, Overdefined };
  explicit SlotValue(State state) : state(state) {}

  State state = State::Unvisited;
  APInt value;
};

/// The forward dataflow of the slot's constant value over a flat region.
///
/// Cleanups need one more fact. FlattenCFG runs a cleanup once for all exits
/// of its scope: each exit stores its own id to the function's cleanup
/// destination slot, and a switch on that slot after the cleanup picks the
/// exit. A goto that leaves the scope is one of those exits, so on a case edge
/// of that switch the goto slot holds what it held where the case's id was
/// stored, even though the paths merged inside the cleanup.
class SlotDataflow {
public:
  SlotDataflow(Region &region, Value slot) : slot(slot) {
    collectCleanupExits(region);
    run(region);
  }

  /// The slot's value when control takes `edge`.
  SlotValue getValueOnEdge(BlockOperand &edge) const;

private:
  void run(Region &region);
  void collectCleanupExits(Region &region);
  SlotValue getValueBefore(Operation *op) const;
  SlotValue getValueAtEnd(Block *block) const;

  Value slot;
  llvm::DenseMap<Block *, SlotValue> entryValues;
  /// The stores of one cleanup id that reach one cleanup dispatch. Unusable
  /// if a path from one of them stores the goto slot.
  struct CleanupExit {
    SmallVector<cir::StoreOp, 2> stores;
    bool usable = true;
  };
  llvm::DenseMap<std::pair<Operation *, int64_t>, CleanupExit> cleanupExits;
  /// Cleanup dispatches reached by a store of an unknown cleanup id, which
  /// could be any of their cases.
  llvm::SmallPtrSet<Operation *, 4> unknownCleanupExits;
};

struct CIRFoldGotoDispatchPass
    : public impl::CIRFoldGotoDispatchBase<CIRFoldGotoDispatchPass> {
  using CIRFoldGotoDispatchBase::CIRFoldGotoDispatchBase;

  void runOnOperation() override;
};

} // namespace

/// Returns the dispatch on `slot` that `block` consists of: a load of the
/// slot followed by a `cir.switch.flat` on the loaded value, and nothing else.
static cir::SwitchFlatOp getDispatch(Block &block, Value slot) {
  if (!llvm::hasNItems(block, 2) || block.getNumArguments() != 0)
    return {};
  auto load = dyn_cast<cir::LoadOp>(block.front());
  auto dispatch = dyn_cast<cir::SwitchFlatOp>(block.back());
  if (!load || !dispatch || load.getAddr() != slot ||
      dispatch.getCondition() != load.getResult())
    return {};
  return dispatch;
}

/// Returns the constant `store` writes, if it writes one.
static std::optional<APInt> getStoredConstant(cir::StoreOp store) {
  auto constant = store.getValue().getDefiningOp<cir::ConstantOp>();
  auto value =
      constant ? dyn_cast<cir::IntAttr>(constant.getValue()) : cir::IntAttr();
  if (!value)
    return std::nullopt;
  return value.getValue();
}

/// Returns the case value that leads `dispatch` along `edge`, if exactly one
/// case and not the default leads there.
static std::optional<APInt> getCaseValueOfEdge(cir::SwitchFlatOp dispatch,
                                               BlockOperand &edge) {
  Block *dest = edge.get();
  if (dispatch.getDefaultDestination() == dest ||
      llvm::count(dispatch.getCaseDestinations(), dest) != 1)
    return std::nullopt;
  for (auto [caseValue, caseDest] :
       llvm::zip_equal(dispatch.getCaseValues().getAsRange<cir::IntAttr>(),
                       dispatch.getCaseDestinations()))
    if (caseDest == dest)
      return caseValue.getValue();
  return std::nullopt;
}

/// Returns the switch that ends `block` if it switches on a value loaded from
/// a `cleanup_dest_slot` alloca in the same block.
static cir::SwitchFlatOp getCleanupDispatch(Block &block) {
  auto dispatch = dyn_cast<cir::SwitchFlatOp>(block.getTerminator());
  if (!dispatch)
    return {};
  auto load = dispatch.getCondition().getDefiningOp<cir::LoadOp>();
  if (!load || load->getBlock() != &block)
    return {};
  auto alloca = load.getAddr().getDefiningOp<cir::AllocaOp>();
  if (!alloca || !alloca.getCleanupDestSlot())
    return {};
  return dispatch;
}

static bool isStoreTo(Operation &op, Value addr) {
  auto store = dyn_cast<cir::StoreOp>(op);
  return store && store.getAddr() == addr;
}

void SlotDataflow::collectCleanupExits(Region &region) {
  // Follow each store of a cleanup id forward until the cleanup slot is
  // written again or a cleanup dispatch reads it.
  region.walk([&](cir::StoreOp exitStore) {
    auto cleanupSlot = exitStore.getAddr().getDefiningOp<cir::AllocaOp>();
    if (!cleanupSlot || !cleanupSlot.getCleanupDestSlot())
      return;
    std::optional<APInt> id = getStoredConstant(exitStore);
    Value cleanupAddr = cleanupSlot.getResult();

    bool writesGotoSlot = false;
    SmallVector<Operation *> reached;
    SmallVector<Operation *> worklist{exitStore->getNextNode()};
    llvm::SmallPtrSet<Block *, 16> visited;
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      Block *block = op->getBlock();
      bool stopped = false;
      for (; op; op = op->getNextNode()) {
        if (isStoreTo(*op, cleanupAddr)) {
          stopped = true;
          break;
        }
        writesGotoSlot |= isStoreTo(*op, slot);
      }
      if (stopped)
        continue;
      if (getCleanupDispatch(*block)) {
        reached.push_back(block->getTerminator());
        continue;
      }
      for (Block *succ : block->getSuccessors())
        if (visited.insert(succ).second)
          worklist.push_back(&succ->front());
    }

    for (Operation *dispatch : reached) {
      if (!id) {
        unknownCleanupExits.insert(dispatch);
        continue;
      }
      CleanupExit &exit = cleanupExits[{dispatch, id->getSExtValue()}];
      exit.stores.push_back(exitStore);
      exit.usable &= !writesGotoSlot;
    }
  });
}

SlotValue SlotDataflow::getValueBefore(Operation *op) const {
  for (Operation *prev = op->getPrevNode(); prev; prev = prev->getPrevNode())
    if (auto store = dyn_cast<cir::StoreOp>(prev);
        store && store.getAddr() == slot) {
      std::optional<APInt> value = getStoredConstant(store);
      return value ? SlotValue::constant(*value) : SlotValue::overdefined();
    }
  return entryValues.lookup(op->getBlock());
}

SlotValue SlotDataflow::getValueAtEnd(Block *block) const {
  return getValueBefore(block->getTerminator());
}

SlotValue SlotDataflow::getValueOnEdge(BlockOperand &edge) const {
  Operation *terminator = edge.getOwner();
  Block *block = terminator->getBlock();
  // A case edge of a dispatch on the slot carries the case's value.
  if (auto dispatch = getDispatch(*block, slot))
    if (std::optional<APInt> caseValue = getCaseValueOfEdge(dispatch, edge))
      return SlotValue::constant(*caseValue);

  // A case edge of a cleanup dispatch carries what the slot held where that
  // case's cleanup id was stored.
  if (auto dispatch = getCleanupDispatch(*block);
      dispatch && !unknownCleanupExits.contains(dispatch)) {
    if (std::optional<APInt> caseValue = getCaseValueOfEdge(dispatch, edge)) {
      auto it = cleanupExits.find({dispatch, caseValue->getSExtValue()});
      if (it != cleanupExits.end() && it->second.usable) {
        SlotValue value;
        for (cir::StoreOp exitStore : it->second.stores)
          value = value.join(getValueBefore(exitStore));
        return value;
      }
    }
  }
  return getValueAtEnd(block);
}

void SlotDataflow::run(Region &region) {
  Block *entry = &region.front();
  entryValues[entry] = SlotValue::overdefined();
  llvm::ReversePostOrderTraversal<Block *> order(entry);
  bool changed = true;
  while (changed) {
    changed = false;
    for (Block *block : order) {
      if (block == entry)
        continue;
      SlotValue value = SlotValue::unvisited();
      for (BlockOperand &use : block->getUses())
        value = value.join(getValueOnEdge(use));
      SlotValue &entryValue = entryValues[block];
      if (entryValue == value)
        continue;
      entryValue = value;
      changed = true;
    }
  }
}

/// Returns the block a dispatch sends `value` to.
static Block *getDispatchDest(cir::SwitchFlatOp dispatch, const APInt &value) {
  for (auto [caseValue, dest] :
       llvm::zip_equal(dispatch.getCaseValues().getAsRange<cir::IntAttr>(),
                       dispatch.getCaseDestinations()))
    if (caseValue.getValue() == value)
      return dest;
  return dispatch.getDefaultDestination();
}

/// Whether the slot is only ever stored constants and loaded from, so that
/// nothing else can change it between a store and a dispatch.
static bool isOnlyLoadedAndStored(cir::AllocaOp slot) {
  return llvm::all_of(slot->getUses(), [&](OpOperand &use) {
    if (auto store = dyn_cast<cir::StoreOp>(use.getOwner()))
      return use.get() == store.getAddr() && !store.getIsVolatile() &&
             getStoredConstant(store);
    return isa<cir::LoadOp>(use.getOwner());
  });
}

/// Whether every operand of every successor of `dispatch` is empty, so an
/// edge can move to any of them unchanged.
static bool hasNoSuccessorOperands(cir::SwitchFlatOp dispatch) {
  return dispatch.getDefaultOperands().empty() &&
         llvm::all_of(dispatch.getCaseOperands(),
                      [](ValueRange operands) { return operands.empty(); });
}

/// Whether `block` only forwards control: a lone `cir.br` without operands.
static bool isForwarder(Block &block) {
  if (block.isEntryBlock() || block.getNumArguments() != 0 ||
      !llvm::hasSingleElement(block))
    return false;
  auto br = dyn_cast<cir::BrOp>(block.front());
  return br && br.getDestOperands().empty();
}

/// Collects the edges that reach `target`, looking through blocks that only
/// forward to it: those carry no store, so the slot's value on an edge into a
/// forwarder is its value on the way into `target`.
static void
collectEdgesThroughForwarders(Block *target,
                              SmallVectorImpl<BlockOperand *> &edges) {
  SmallVector<Block *> worklist{target};
  llvm::SmallPtrSet<Block *, 8> seen{target};
  while (!worklist.empty()) {
    for (BlockOperand &use : worklist.pop_back_val()->getUses()) {
      Block *pred = use.getOwner()->getBlock();
      if (!isForwarder(*pred)) {
        edges.push_back(&use);
        continue;
      }
      if (seen.insert(pred).second)
        worklist.push_back(pred);
    }
  }
}

/// Redirects every edge into a dispatch whose slot value is known. Returns
/// whether any edge moved.
static bool threadDispatches(cir::FuncOp func, Value slot, IRRewriter &rewriter,
                             unsigned &numThreaded) {
  Region &body = func.getBody();
  SlotDataflow dataflow(body, slot);
  bool changed = false;
  for (Block &block : body) {
    cir::SwitchFlatOp dispatch = getDispatch(block, slot);
    if (!dispatch || !hasNoSuccessorOperands(dispatch))
      continue;
    SmallVector<BlockOperand *> edges;
    collectEdgesThroughForwarders(&block, edges);
    for (BlockOperand *edge : edges) {
      std::optional<APInt> value = dataflow.getValueOnEdge(*edge).getConstant();
      if (!value)
        continue;
      Block *dest = getDispatchDest(dispatch, *value);
      if (dest == edge->get() || dest->getNumArguments() != 0)
        continue;
      rewriter.modifyOpInPlace(edge->getOwner(), [&] { edge->set(dest); });
      ++numThreaded;
      changed = true;
    }
  }
  return changed;
}

void CIRFoldGotoDispatchPass::runOnOperation() {
  cir::FuncOp func = getOperation();
  if (func.getBody().empty())
    return;
  Block &entry = func.getBody().front();
  auto allocas = entry.getOps<cir::AllocaOp>();
  auto slotIt = llvm::find_if(
      allocas, [](cir::AllocaOp op) { return op.getGotoDestSlot(); });
  if (slotIt == allocas.end() || !isOnlyLoadedAndStored(*slotIt))
    return;
  cir::AllocaOp slot = *slotIt;

  IRRewriter rewriter(&getContext());
  unsigned numThreaded = 0;
  // A thread can make a later edge known, for instance an edge that now
  // leaves a case of an earlier dispatch directly.
  while (threadDispatches(func, slot, rewriter, numThreaded)) {
    // Blocks left without predecessors would otherwise keep their stale
    // values in the dataflow and their loads of the slot; drop them, and the
    // blocks only they reached, before the next round. A dead block may
    // still use a value of another one, so drop all of their uses before
    // erasing any.
    while (true) {
      SmallVector<Block *> dead;
      for (Block &block : func.getBody())
        if (&block != &entry && block.hasNoPredecessors() &&
            !isa<cir::LabelOp>(block.front()))
          dead.push_back(&block);
      if (dead.empty())
        break;
      for (Block *block : dead)
        block->dropAllDefinedValueUses();
      for (Block *block : dead)
        rewriter.eraseBlock(block);
    }
  }
  numThreadedEdges += numThreaded;

  // Once no dispatch reads the slot, its stores are dead too.
  if (llvm::any_of(slot->getUsers(),
                   [](Operation *user) { return isa<cir::LoadOp>(user); }))
    return;
  for (Operation *user : llvm::make_early_inc_range(slot->getUsers())) {
    Value stored = cast<cir::StoreOp>(user).getValue();
    rewriter.eraseOp(user);
    if (Operation *def = stored.getDefiningOp(); def && def->use_empty())
      rewriter.eraseOp(def);
  }
  rewriter.eraseOp(slot);
  ++numErasedSlots;
}

std::unique_ptr<Pass> mlir::createCIRFoldGotoDispatchPass() {
  return std::make_unique<CIRFoldGotoDispatchPass>();
}
