//===-- CSE.cpp -- common subexpression elimination -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This transformation pass performs a simple common sub-expression elimination
/// algorithm on operations within a function.
///
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/RecyclingAllocator.h"
#include <deque>

using namespace mlir;

static llvm::cl::opt<bool>
    leaveEffects("keep-effects",
                 llvm::cl::desc("disable cleaning up effects attributes"),
                 llvm::cl::init(false), llvm::cl::Hidden);
static llvm::cl::opt<bool> disableCSE("disable-cse",
                                      llvm::cl::desc("disable CSE pass"),
                                      llvm::cl::init(false), llvm::cl::Hidden);

namespace {

struct SimpleOperationInfo : public llvm::DenseMapInfo<Operation *> {

  /// Compute the hash value of an Operation
  static unsigned getHashValue(const Operation *opC) {
    auto *op = const_cast<Operation *>(opC);
    // Hash the operations based upon their:
    //   - Operation Name
    //   - Attributes
    //   - Result Types
    //   - Operands
    unsigned hashOps;
    if (op->isCommutative()) {
      std::vector<void *> vec;
      for (auto i = op->operand_begin(), e = op->operand_end(); i != e; ++i)
        vec.push_back((*i).getAsOpaquePointer());
      llvm::sort(vec.begin(), vec.end());
      hashOps = llvm::hash_combine_range(vec.begin(), vec.end());
    } else {
      hashOps =
          llvm::hash_combine_range(op->operand_begin(), op->operand_end());
    }
    auto hashResTys{llvm::hash_combine_range(op->result_type_begin(),
                                             op->result_type_end())};
    return llvm::hash_combine(op->getName(), op->getAttrs(), hashResTys,
                              hashOps);
  }

  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;

    // Compare the operation name.
    if (lhs->getName() != rhs->getName())
      return false;
    // Check operand and result type counts.
    if (lhs->getNumOperands() != rhs->getNumOperands() ||
        lhs->getNumResults() != rhs->getNumResults())
      return false;
    // Compare attributes.
    if (lhs->getAttrs() != rhs->getAttrs())
      return false;
    // Compare operands.
    if (lhs->isCommutative()) {
      SmallVector<void*, 8> lops;
      for (const auto &lod : lhs->getOperands())
        lops.push_back(lod.getAsOpaquePointer());
      llvm::sort(lops.begin(), lops.end());
      SmallVector<void*, 8> rops;
      for (const auto &rod : rhs->getOperands())
        rops.push_back(rod.getAsOpaquePointer());
      llvm::sort(rops.begin(), rops.end());
      if (!std::equal(lops.begin(), lops.end(), rops.begin()))
        return false;
    } else {
      if (!std::equal(lhs->operand_begin(), lhs->operand_end(),
                      rhs->operand_begin()))
        return false;
    }
    // Compare result types.
    return std::equal(lhs->result_type_begin(), lhs->result_type_end(),
                      rhs->result_type_begin());
  }
};

/// Basic common sub-expression elimination.
struct BasicCSE : public mlir::PassWrapper<BasicCSE, mlir::FunctionPass> {
  BasicCSE() {}
  BasicCSE(const BasicCSE &) {}

  /// Shared implementation of operation elimination and scoped map definitions.
  using AllocatorTy = llvm::RecyclingAllocator<
      llvm::BumpPtrAllocator,
      llvm::ScopedHashTableVal<Operation *, Operation *>>;
  using ScopedMapTy = llvm::ScopedHashTable<Operation *, Operation *,
                                            SimpleOperationInfo, AllocatorTy>;

  /// Represents a single entry in the depth first traversal of a CFG.
  struct CFGStackNode {
    CFGStackNode(ScopedMapTy &knownValues, DominanceInfoNode *node)
        : scope(knownValues), node(node), childIterator(node->begin()),
          processed(false) {}

    /// Scope for the known values.
    ScopedMapTy::ScopeTy scope;

    DominanceInfoNode *node;
    DominanceInfoNode::iterator childIterator;

    /// If this node has been fully processed yet or not.
    bool processed;
  };

  /// Attempt to eliminate a redundant operation. Returns success if the
  /// operation was marked for removal, failure otherwise.
  LogicalResult simplifyOperation(ScopedMapTy &knownValues, Operation *op);

  void simplifyBlock(ScopedMapTy &knownValues, DominanceInfo &domInfo,
                     Block *bb);
  void simplifyRegion(ScopedMapTy &knownValues, DominanceInfo &domInfo,
                      Region &region);

  void cleanupBlock(Block *bb) {
    for (auto &inst : *bb) {
      if (fir::nonVolatileLoad(&inst) || fir::pureCall(&inst)) {
        inst.removeAttr(Identifier::get("effects_token", inst.getContext()));
      } else if (inst.getNumRegions()) {
        for (auto &region : inst.getRegions())
          cleanupRegion(region);
      }
    }
  }
  void cleanupRegion(Region &region) {
    for (auto &block : region)
      cleanupBlock(&block);
  }

  void runOnFunction() override final;

private:
  /// Operations marked as dead and to be erased.
  std::vector<Operation *> opsToErase;
};

/// Attempt to eliminate a redundant operation.
LogicalResult BasicCSE::simplifyOperation(ScopedMapTy &knownValues,
                                          Operation *op) {
  // Don't simplify operations with nested blocks. We don't currently model
  // equality comparisons correctly among other things. It is also unclear
  // whether we would want to CSE such operations.
  if (op->getNumRegions() != 0)
    return failure();

  if (!MemoryEffectOpInterface::hasNoEffect(op) && !fir::nonVolatileLoad(op) &&
      !fir::pureCall(op))
    return failure();

  // If the operation is already trivially dead just add it to the erase list.
  if (isOpTriviallyDead(op)) {
    opsToErase.push_back(op);
    return success();
  }

  // Look for an existing definition for the operation.
  if (auto *existing = knownValues.lookup(op)) {
    // If we find one then replace all uses of the current operation with the
    // existing one and mark it for deletion.
    op->replaceAllUsesWith(existing);
    if (op->isKnownNonTerminator())
      opsToErase.push_back(op);

    // If the existing operation has an unknown location and the current
    // operation doesn't, then set the existing op's location to that of the
    // current op.
    if (existing->getLoc().isa<UnknownLoc>() &&
        !op->getLoc().isa<UnknownLoc>()) {
      existing->setLoc(op->getLoc());
    }
    return success();
  }

  // Otherwise, we add this operation to the known values map.
  knownValues.insert(op, op);
  return failure();
}

void BasicCSE::simplifyBlock(ScopedMapTy &knownValues, DominanceInfo &domInfo,
                             Block *bb) {
  std::intptr_t token = reinterpret_cast<std::intptr_t>(bb);
  for (auto &inst : *bb) {
    if (fir::nonVolatileLoad(&inst) || fir::pureCall(&inst))
      inst.setAttr("effects_token",
                   IntegerAttr::get(IndexType::get(inst.getContext()), token));
    if (dyn_cast<fir::StoreOp>(&inst) || fir::impureCall(&inst))
      token = reinterpret_cast<std::intptr_t>(&inst);
  }
  for (auto &inst : *bb) {
    // If the operation is simplified, we don't process any held regions.
    if (succeeded(simplifyOperation(knownValues, &inst)))
      continue;

    // If this operation is isolated above, we can't process nested regions with
    // the given 'knownValues' map. This would cause the insertion of implicit
    // captures in explicit capture only regions.
    if (!inst.isRegistered() || inst.isKnownIsolatedFromAbove()) {
      ScopedMapTy nestedKnownValues;
      for (auto &region : inst.getRegions())
        simplifyRegion(nestedKnownValues, domInfo, region);
      continue;
    }

    // Otherwise, process nested regions normally.
    for (auto &region : inst.getRegions())
      simplifyRegion(knownValues, domInfo, region);
  }
}

void BasicCSE::simplifyRegion(ScopedMapTy &knownValues, DominanceInfo &domInfo,
                              Region &region) {
  // If the region is empty there is nothing to do.
  if (region.empty())
    return;

  // If the region only contains one block, then simplify it directly.
  if (std::next(region.begin()) == region.end()) {
    ScopedMapTy::ScopeTy scope(knownValues);
    simplifyBlock(knownValues, domInfo, &region.front());
    return;
  }

  // Note, deque is being used here because there was significant performance
  // gains over vector when the container becomes very large due to the
  // specific access patterns. If/when these performance issues are no
  // longer a problem we can change this to vector. For more information see
  // the llvm mailing list discussion on this:
  // http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20120116/135228.html
  std::deque<std::unique_ptr<CFGStackNode>> stack;

  // Process the nodes of the dom tree for this region.
  stack.emplace_back(std::make_unique<CFGStackNode>(
      knownValues, domInfo.getRootNode(&region)));

  while (!stack.empty()) {
    auto &currentNode = stack.back();

    // Check to see if we need to process this node.
    if (!currentNode->processed) {
      currentNode->processed = true;
      simplifyBlock(knownValues, domInfo, currentNode->node->getBlock());
    }

    // Otherwise, check to see if we need to process a child node.
    if (currentNode->childIterator != currentNode->node->end()) {
      auto *childNode = *(currentNode->childIterator++);
      stack.emplace_back(
          std::make_unique<CFGStackNode>(knownValues, childNode));
    } else {
      // Finally, if the node and all of its children have been processed
      // then we delete the node.
      stack.pop_back();
    }
  }
}

void BasicCSE::runOnFunction() {
  if (disableCSE)
    return;

  /// A scoped hash table of defining operations within a function.
  {
    ScopedMapTy knownValues;
    simplifyRegion(knownValues, getAnalysis<DominanceInfo>(),
                   getFunction().getBody());
  }
  if (!leaveEffects) {
    cleanupRegion(getFunction().getBody());
  }

  // If no operations were erased, then we mark all analyses as preserved.
  if (opsToErase.empty())
    return markAllAnalysesPreserved();

  /// Erase any operations that were marked as dead during simplification.
  for (auto *op : opsToErase)
    op->erase();
  opsToErase.clear();

  // We currently don't remove region operations, so mark dominance as
  // preserved.
  markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
}

} // end anonymous namespace

std::unique_ptr<mlir::Pass> fir::createCSEPass() {
  return std::make_unique<BasicCSE>();
}
