//===- GotoAnalysis.cpp - Reachability through jumps in CIR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/GotoAnalysis.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace cir;

/// Return whether control enters `op` through a jump that is not a CFG edge.
static bool isJumpTarget(Operation *op) {
  return isa<cir::LabelOp, cir::CaseOp>(op);
}

/// Return whether `op` may define the successors of its block.
static bool mayDefineSuccessors(Operation *op) {
  return op->mightHaveTrait<OpTrait::IsTerminator>() ||
         op->getNumSuccessors() != 0;
}

/// Return the distinct successors of `block` in its own region. A successor in
/// another region is invalid IR that cannot make a block of this region
/// reachable.
static SmallVector<Block *, 2> getRegionSuccessors(Block *block) {
  SmallVector<Block *, 2> successors;
  for (Block *successor : block->getSuccessors())
    if (successor->getParent() == block->getParent() &&
        !llvm::is_contained(successors, successor))
      successors.push_back(successor);
  return successors;
}

GotoAnalysis::GotoAnalysis(Operation *root) : root(root) {
  root->walk([&](Operation *op) {
    if (isJumpTarget(op))
      addHolders(op);
  });
}

void GotoAnalysis::addHolders(Operation *target) {
  // A goto may come from anywhere in the function, but only the switch of a
  // case jumps to it, so its holders end below that switch.
  Operation *source = nullptr;
  if (isa<cir::CaseOp>(target))
    source = target->getParentOfType<cir::SwitchOp>();
  for (Operation *op = target; op != root && op != source;
       op = op->getParentOp()) {
    if (holders.insert(op).second)
      markGained(op);
    Operation *parent = op->getParentOp();
    if (!parent || parent->hasTrait<OpTrait::IsIsolatedFromAbove>())
      return;
  }
}

void GotoAnalysis::addNestedHolders(Operation *op) {
  op->walk([&](Operation *nested) {
    if (isJumpTarget(nested))
      addHolders(nested);
  });
}

void GotoAnalysis::reach(RegionState &state, Block *block, Operation *entry) {
  SmallVector<std::pair<Block *, Operation *>> worklist{{block, entry}};
  while (!worklist.empty()) {
    auto [current, currentEntry] = worklist.pop_back_val();
    auto [it, inserted] = state.reached.try_emplace(current);
    if (!inserted) {
      // An edge reaches the whole of a block entered at a jump target.
      if (!currentEntry)
        it->second.entry = nullptr;
      continue;
    }
    it->second.entry = currentEntry;
    it->second.successors = getRegionSuccessors(current);
    for (Block *successor : it->second.successors)
      worklist.emplace_back(successor, nullptr);
  }
}

void GotoAnalysis::compute(Region *region, RegionState &state) {
  state.stale = false;
  state.entry = &region->front();
  state.reached.clear();
  state.changed.clear();
  state.removed.clear();
  state.gained.clear();
  state.lost.clear();
  reach(state, state.entry, nullptr);
  // Control also enters a block at its first jump target or holder of one.
  for (Block &block : llvm::drop_begin(*region)) {
    auto it = state.reached.find(&block);
    if (it != state.reached.end() && !it->second.entry)
      continue;
    auto entry = llvm::find_if(
        block, [&](Operation &op) { return holders.contains(&op); });
    if (entry != block.end())
      reach(state, &block, &*entry);
  }
}

LogicalResult GotoAnalysis::update(Region *region, RegionState &state) {
  for (Block *block : state.changed) {
    // A change in a block that left the region or that control does not reach
    // reaches nothing new.
    if (state.removed.contains(block))
      continue;
    auto it = state.reached.find(block);
    if (it == state.reached.end())
      continue;
    SmallVector<Block *, 2> current = getRegionSuccessors(block);
    for (Block *lost : it->second.successors) {
      if (llvm::is_contained(current, lost))
        continue;
      // Merging `lost` into `block` erases `lost` and gives its successors to
      // `block`, which changes no other block. Any other lost edge may
      // disconnect blocks.
      auto lostIt = state.reached.find(lost);
      if (!state.removed.contains(lost) || lostIt == state.reached.end() ||
          !llvm::all_of(lostIt->second.successors, [&](Block *successor) {
            return llvm::is_contained(current, successor);
          }))
        return failure();
    }
    it->second.successors = current;
    // Added edges can only make more blocks reachable.
    for (Block *successor : current)
      reach(state, successor, nullptr);
  }
  // A holder left a block that control reaches.
  for (auto [block, holder] : state.lost) {
    if (block == &region->front())
      continue;
    auto it = state.reached.find(block);
    if (it == state.reached.end())
      continue;
    // The block left the region, and the blocks it reached may have gone with
    // it. Merging it into another block is fine: the holder and the successors
    // move to that block. A holder erased since is no longer in `holders`, so
    // it is not dereferenced.
    if (state.removed.contains(block)) {
      if (it->second.successors.empty())
        continue;
      if (!holders.contains(holder) || holder->getParentRegion() != region)
        return failure();
      SmallVector<Block *, 2> successors =
          getRegionSuccessors(holder->getBlock());
      if (!llvm::all_of(it->second.successors, [&](Block *successor) {
            return llvm::is_contained(successors, successor);
          }))
        return failure();
      continue;
    }
    Operation *entry = it->second.entry;
    // An earlier holder still lets control in.
    if (entry && entry != holder)
      continue;
    // An edge reaches the whole block, but it may come around a cycle that
    // control entered at `holder`.
    if (!entry)
      return failure();
    // Control entered the block only at `holder`, so it now enters at the next
    // holder. Without one, the blocks it reached may be disconnected.
    auto next = llvm::find_if(
        *block, [&](Operation &op) { return holders.contains(&op); });
    if (next == block->end())
      return failure();
    it->second.entry = &*next;
  }
  // Control also enters a block at a holder that arrived in it. A holder
  // erased since is no longer in `holders`, so it is not dereferenced.
  for (Operation *holder : state.gained) {
    if (!holders.contains(holder) || holder->getParentRegion() != region)
      continue;
    Block *block = holder->getBlock();
    auto it = state.reached.find(block);
    if (it == state.reached.end())
      reach(state, block, holder);
    else if (Operation *entry = it->second.entry;
             entry && holder->isBeforeInBlock(entry))
      it->second.entry = holder;
  }
  // Drop the blocks that left the region only now, so that the successors of a
  // lost block remain available to every block that lost an edge to it.
  for (Block *block : state.removed)
    state.reached.erase(block);
  state.changed.clear();
  state.removed.clear();
  state.gained.clear();
  state.lost.clear();
  return success();
}

GotoAnalysis::RegionState &GotoAnalysis::getState(Region *region) {
  RegionState &state = regions[region];
  if (state.stale || state.entry != &region->front() ||
      failed(update(region, state)))
    compute(region, state);
  return state;
}

bool GotoAnalysis::isReachable(Operation *op) {
  for (; op != root; op = op->getParentOp()) {
    assert(op && "operation is not nested in the analyzed root");
    Block *block = op->getBlock();
    Region *region = block ? block->getParent() : nullptr;
    // A detached operation counts as reachable, as does an entry block.
    if (!region)
      return true;
    if (block == &region->front())
      continue;
    RegionState &state = getState(region);
    auto it = state.reached.find(block);
    if (it == state.reached.end())
      return false;
    if (Operation *entry = it->second.entry;
        entry && op != entry && op->isBeforeInBlock(entry))
      return false;
  }
  return true;
}

void GotoAnalysis::markChanged(Block *block) {
  if (!block)
    return;
  auto it = regions.find(block->getParent());
  if (it != regions.end())
    it->second.changed.insert(block);
}

void GotoAnalysis::markRemoved(Region *region, Block *block) {
  auto it = regions.find(region);
  if (it != regions.end())
    it->second.removed.insert(block);
}

void GotoAnalysis::markGained(Operation *holder) {
  auto it = regions.find(holder->getParentRegion());
  if (it != regions.end())
    it->second.gained.push_back(holder);
}

void GotoAnalysis::markLost(Block *block, Operation *holder) {
  if (!block)
    return;
  auto it = regions.find(block->getParent());
  if (it != regions.end())
    it->second.lost.emplace_back(block, holder);
}

void GotoAnalysis::markStale(Region *region) {
  auto it = regions.find(region);
  if (it != regions.end())
    it->second.stale = true;
}

void GotoAnalysis::notifyOperationInserted(Operation *op,
                                           OpBuilder::InsertPoint previous) {
  Block *from = previous.isSet() ? previous.getBlock() : nullptr;
  // A new operation may hold jump targets that were inserted into its regions
  // before the regions belonged to it. A holder that moves to another parent
  // gives the jump targets it holds new holders. The old ones keep their mark,
  // which only over-approximates.
  if (from ? holders.contains(op) && from->getParentOp() != op->getParentOp()
           : isJumpTarget(op) || op->getNumRegions() != 0)
    addNestedHolders(op);
  if (holders.contains(op)) {
    markLost(from, op);
    markGained(op);
  }
  if (mayDefineSuccessors(op)) {
    markChanged(op->getBlock());
    markChanged(from);
  }
}

void GotoAnalysis::notifyOperationModified(Operation *op) {
  if (mayDefineSuccessors(op))
    markChanged(op->getBlock());
}

void GotoAnalysis::notifyOperationErased(Operation *op) {
  // Drop the regions that go away with `op`, so that a region allocated at the
  // same address later does not find their state.
  for (Region &region : op->getRegions())
    regions.erase(&region);
  if (holders.erase(op))
    markLost(op->getBlock(), op);
  if (mayDefineSuccessors(op))
    markChanged(op->getBlock());
}

void GotoAnalysis::notifyBlockInserted(Block *block, Region *previous,
                                       Region::iterator) {
  Region *region = block->getParent();
  if (previous == region)
    return;
  // A block new to the region is unreachable until an edge or a jump target
  // reaches it, but a state at its address left by an earlier block cannot be
  // trusted.
  if (previous)
    markRemoved(previous, block);
  auto it = regions.find(region);
  if (it != regions.end() && (it->second.removed.contains(block) ||
                              it->second.reached.contains(block)))
    it->second.stale = true;
  // The jump targets in a moved block have new holders, and control enters
  // the block at them.
  if (previous && llvm::any_of(*block, [&](Operation &op) {
        return holders.contains(&op);
      })) {
    for (Operation &op : *block)
      addNestedHolders(&op);
    markStale(region);
    markStale(previous);
  }
}

void GotoAnalysis::notifyBlockErased(Block *block) {
  markRemoved(block->getParent(), block);
}
