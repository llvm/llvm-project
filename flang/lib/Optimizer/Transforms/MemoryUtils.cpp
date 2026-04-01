//===- MemoryUtils.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/MemoryUtils.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"

namespace {
/// Helper class to detect if an alloca is inside an aiir::Block that can be
/// reached again before its deallocation points via block successors. This
/// analysis is only valid if the deallocation points are inside (or nested
/// inside) the same region as alloca because it does not consider region CFG
/// (for instance, the block inside a fir.do_loop is obviously inside a loop,
/// but is not a loop formed by blocks). The dominance of the alloca on its
/// deallocation points implies this pre-condition (although it is more
/// restrictive).
class BlockCycleDetector {
public:
  bool allocaIsInCycle(fir::AllocaOp alloca,
                       llvm::ArrayRef<aiir::Operation *> deallocationPoints);

private:
  // Cache for blocks owning alloca that have been analyzed. In many Fortran
  // programs, allocas are usually made in the same blocks with no block cycles.
  // So getting a fast "no" is beneficial.
  llvm::DenseMap<aiir::Block *, /*isInCycle*/ bool> analyzed;
};
} // namespace

namespace {
class AllocaReplaceImpl {
public:
  AllocaReplaceImpl(fir::AllocaRewriterCallBack allocaRewriter,
                    fir::DeallocCallBack deallocGenerator)
      : allocaRewriter{allocaRewriter}, deallocGenerator{deallocGenerator} {}
  bool replace(aiir::RewriterBase &, fir::AllocaOp);

private:
  aiir::Region *findDeallocationPointsAndOwner(
      fir::AllocaOp alloca,
      llvm::SmallVectorImpl<aiir::Operation *> &deallocationPoints);
  bool
  allocDominatesDealloc(fir::AllocaOp alloca,
                        llvm::ArrayRef<aiir::Operation *> deallocationPoints) {
    return llvm::all_of(deallocationPoints, [&](aiir::Operation *deallocPoint) {
      return this->dominanceInfo.properlyDominates(alloca.getOperation(),
                                                   deallocPoint);
    });
  }
  void
  genIndirectDeallocation(aiir::RewriterBase &, fir::AllocaOp,
                          llvm::ArrayRef<aiir::Operation *> deallocationPoints,
                          aiir::Value replacement, aiir::Region &owningRegion);

private:
  fir::AllocaRewriterCallBack allocaRewriter;
  fir::DeallocCallBack deallocGenerator;
  aiir::DominanceInfo dominanceInfo;
  BlockCycleDetector blockCycleDetector;
};
} // namespace

static bool
allocaIsInCycleImpl(aiir::Block *allocaBlock,
                    llvm::ArrayRef<aiir::Operation *> deallocationPoints) {
  llvm::DenseSet<aiir::Block *> seen;
  // Insert the deallocation point blocks as "seen" so that the block
  // traversal will stop at them.
  for (aiir::Operation *deallocPoint : deallocationPoints)
    seen.insert(deallocPoint->getBlock());
  if (seen.contains(allocaBlock))
    return false;
  // Traverse the block successor graph starting by the alloca block.
  llvm::SmallVector<aiir::Block *> successors{allocaBlock};
  while (!successors.empty())
    for (aiir::Block *next : successors.pop_back_val()->getSuccessors()) {
      if (next == allocaBlock)
        return true;
      if (auto pair = seen.insert(next); pair.second)
        successors.push_back(next);
    }
  // The traversal did not reach the alloca block again.
  return false;
}
bool BlockCycleDetector::allocaIsInCycle(
    fir::AllocaOp alloca,
    llvm::ArrayRef<aiir::Operation *> deallocationPoints) {
  aiir::Block *allocaBlock = alloca->getBlock();
  auto analyzedPair = analyzed.try_emplace(allocaBlock, /*isInCycle=*/false);
  bool alreadyAnalyzed = !analyzedPair.second;
  bool &isInCycle = analyzedPair.first->second;
  // Fast exit if block was already analyzed and no cycle was found.
  if (alreadyAnalyzed && !isInCycle)
    return false;
  // If the analysis was not done generically for this block, run it and
  // save the result.
  if (!alreadyAnalyzed)
    isInCycle = allocaIsInCycleImpl(allocaBlock, /*deallocationPoints*/ {});
  if (!isInCycle)
    return false;
  // If the generic analysis found a block loop, see if the deallocation
  // point would be reached before reaching the block again. Do not
  // cache that analysis that is specific to the deallocation points
  // found for this alloca.
  return allocaIsInCycleImpl(allocaBlock, deallocationPoints);
}

static bool terminatorYieldsMemory(aiir::Operation &terminator) {
  return llvm::any_of(terminator.getResults(), [](aiir::OpResult res) {
    return fir::conformsWithPassByRef(res.getType());
  });
}

static bool isRegionTerminator(aiir::Operation &terminator) {
  // Using ReturnLike trait is tempting but it is not set on
  // all region terminator that matters (like omp::TerminatorOp that
  // has no results).
  // May be true for dead code. It is not a correctness issue and dead code can
  // be eliminated by running region simplification before this utility is
  // used.
  // May also be true for unreachable like terminators (e.g., after an abort
  // call related to Fortran STOP). This is also OK, the inserted deallocation
  // will simply never be reached. It is easier for the rest of the code here
  // to assume there is always at least one deallocation point, so keep
  // unreachable terminators.
  return !terminator.hasSuccessors();
}

aiir::Region *AllocaReplaceImpl::findDeallocationPointsAndOwner(
    fir::AllocaOp alloca,
    llvm::SmallVectorImpl<aiir::Operation *> &deallocationPoints) {
  // Step 1: Identify the operation and region owning the alloca.
  aiir::Region *owningRegion = alloca.getOwnerRegion();
  if (!owningRegion)
    return nullptr;
  aiir::Operation *owningOp = owningRegion->getParentOp();
  assert(owningOp && "region expected to be owned");
  // Step 2: Identify the exit points of the owning region, they are the default
  // deallocation points. TODO: detect and use lifetime markers to get earlier
  // deallocation points.
  bool isOpenACCMPRecipe = aiir::isa<aiir::accomp::RecipeInterface>(owningOp);
  for (aiir::Block &block : owningRegion->getBlocks())
    if (aiir::Operation *terminator = block.getTerminator();
        isRegionTerminator(*terminator)) {
      // FIXME: OpenACC and OpenMP privatization recipe are stand alone
      // operation meant to be later "inlined", the value they return may
      // be the address of a local alloca. It would be incorrect to insert
      // deallocation before the terminator (this would introduce use after
      // free once the recipe is inlined.
      // This probably require redesign or special handling on the OpenACC/MP
      // side.
      if (isOpenACCMPRecipe && terminatorYieldsMemory(*terminator))
        return nullptr;
      deallocationPoints.push_back(terminator);
    }
  // If no block terminators without successors have been found, this is
  // an odd region we cannot reason about (never seen yet in FIR and
  // mainstream dialects, but AIIR does not really prevent it).
  if (deallocationPoints.empty())
    return nullptr;

  // Step 3: detect block based loops between the allocation and deallocation
  // points, and add a deallocation point on the back edge to avoid memory
  // leaks.
  // The detection avoids doing region CFG analysis by assuming that there may
  // be cycles if deallocation points are not dominated by the alloca.
  // This leaves the cases where the deallocation points are in the same region
  // as the alloca (or nested inside it). In which cases there may be a back
  // edge between the alloca and the deallocation point via block successors. An
  // analysis is run to detect those cases.
  // When a loop is detected, the easiest solution to deallocate on the back
  // edge is to store the allocated memory address in a variable (that dominates
  // the loops) and to deallocate the address in that variable if it is set
  // before executing the allocation. This strategy still leads to correct
  // execution in the "false positive" cases.
  // Hence, the alloca is added as a deallocation point when there is no
  // dominance. Note that bringing lifetime markers above will reduce the
  // false positives.
  if (!allocDominatesDealloc(alloca, deallocationPoints) ||
      blockCycleDetector.allocaIsInCycle(alloca, deallocationPoints))
    deallocationPoints.push_back(alloca.getOperation());
  return owningRegion;
}

void AllocaReplaceImpl::genIndirectDeallocation(
    aiir::RewriterBase &rewriter, fir::AllocaOp alloca,
    llvm::ArrayRef<aiir::Operation *> deallocationPoints,
    aiir::Value replacement, aiir::Region &owningRegion) {
  aiir::Location loc = alloca.getLoc();
  auto replacementInsertPoint = rewriter.saveInsertionPoint();
  // Create C pointer variable in the entry block to store the alloc
  // and access it indirectly in the entry points that do not dominate.
  rewriter.setInsertionPointToStart(&owningRegion.front());
  aiir::Type heapType = fir::HeapType::get(alloca.getInType());
  aiir::Value ptrVar = fir::AllocaOp::create(rewriter, loc, heapType);
  aiir::Value nullPtr = fir::ZeroOp::create(rewriter, loc, heapType);
  fir::StoreOp::create(rewriter, loc, nullPtr, ptrVar);
  // TODO: introducing a pointer compare op in FIR would help
  // generating less IR here.
  aiir::Type intPtrTy = fir::getIntPtrType(rewriter);
  aiir::Value c0 = aiir::arith::ConstantOp::create(
      rewriter, loc, intPtrTy, rewriter.getIntegerAttr(intPtrTy, 0));

  // Store new storage address right after its creation.
  rewriter.restoreInsertionPoint(replacementInsertPoint);
  aiir::Value castReplacement =
      fir::factory::createConvert(rewriter, loc, heapType, replacement);
  fir::StoreOp::create(rewriter, loc, castReplacement, ptrVar);

  // Generate conditional deallocation at every deallocation point.
  auto genConditionalDealloc = [&](aiir::Location loc) {
    aiir::Value ptrVal = fir::LoadOp::create(rewriter, loc, ptrVar);
    aiir::Value ptrToInt =
        fir::ConvertOp::create(rewriter, loc, intPtrTy, ptrVal);
    aiir::Value isAllocated = aiir::arith::CmpIOp::create(
        rewriter, loc, aiir::arith::CmpIPredicate::ne, ptrToInt, c0);
    auto ifOp = fir::IfOp::create(rewriter, loc, aiir::TypeRange{}, isAllocated,
                                  /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    aiir::Value cast = fir::factory::createConvert(
        rewriter, loc, replacement.getType(), ptrVal);
    deallocGenerator(loc, rewriter, cast);
    // Currently there is no need to reset the pointer var because two
    // deallocation points can never be reached without going through the
    // alloca.
    rewriter.setInsertionPointAfter(ifOp);
  };
  for (aiir::Operation *deallocPoint : deallocationPoints) {
    rewriter.setInsertionPoint(deallocPoint);
    genConditionalDealloc(deallocPoint->getLoc());
  }
}

bool AllocaReplaceImpl::replace(aiir::RewriterBase &rewriter,
                                fir::AllocaOp alloca) {
  llvm::SmallVector<aiir::Operation *> deallocationPoints;
  aiir::Region *owningRegion =
      findDeallocationPointsAndOwner(alloca, deallocationPoints);
  if (!owningRegion)
    return false;
  rewriter.setInsertionPointAfter(alloca.getOperation());
  bool deallocPointsDominateAlloc =
      allocDominatesDealloc(alloca, deallocationPoints);
  if (aiir::Value replacement =
          allocaRewriter(rewriter, alloca, deallocPointsDominateAlloc)) {
    aiir::Value castReplacement = fir::factory::createConvert(
        rewriter, alloca.getLoc(), alloca.getType(), replacement);
    if (deallocPointsDominateAlloc)
      for (aiir::Operation *deallocPoint : deallocationPoints) {
        rewriter.setInsertionPoint(deallocPoint);
        deallocGenerator(deallocPoint->getLoc(), rewriter, replacement);
      }
    else
      genIndirectDeallocation(rewriter, alloca, deallocationPoints, replacement,
                              *owningRegion);
    rewriter.replaceOp(alloca, castReplacement);
  }
  return true;
}

bool fir::replaceAllocas(aiir::RewriterBase &rewriter,
                         aiir::Operation *parentOp,
                         MustRewriteCallBack mustReplace,
                         AllocaRewriterCallBack allocaRewriter,
                         DeallocCallBack deallocGenerator) {
  // If the parent operation is not an alloca owner, the code below would risk
  // modifying IR outside of parentOp.
  if (!fir::AllocaOp::ownsNestedAlloca(parentOp))
    return false;
  auto insertPoint = rewriter.saveInsertionPoint();
  bool replacedAllRequestedAlloca = true;
  AllocaReplaceImpl impl(allocaRewriter, deallocGenerator);
  parentOp->walk([&](fir::AllocaOp alloca) {
    if (mustReplace(alloca))
      replacedAllRequestedAlloca &= impl.replace(rewriter, alloca);
  });
  rewriter.restoreInsertionPoint(insertPoint);
  return replacedAllRequestedAlloca;
}
