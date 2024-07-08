//===- MemoryUtils.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/MemoryUtils.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"

namespace {
class AllocaReplaceImpl {
public:
  AllocaReplaceImpl(fir::AllocaRewriterCallBack allocaRewriter,
                    fir::DeallocCallBack deallocGenerator)
      : allocaRewriter{allocaRewriter}, deallocGenerator{deallocGenerator} {}
  bool replace(mlir::RewriterBase &, fir::AllocaOp);

private:
  mlir::Region *findDeallocationPointsAndOwner(
      fir::AllocaOp alloca,
      llvm::SmallVectorImpl<mlir::Operation *> &deallocationPoints);
  bool
  allocDominatesDealloc(fir::AllocaOp alloca,
                        llvm::ArrayRef<mlir::Operation *> deallocationPoints) {
    return llvm::all_of(deallocationPoints, [&](mlir::Operation *deallocPoint) {
      return this->dominanceInfo.properlyDominates(alloca.getOperation(),
                                                   deallocPoint);
    });
  }
  void
  genIndirectDeallocation(mlir::RewriterBase &, fir::AllocaOp,
                          llvm::ArrayRef<mlir::Operation *> deallocationPoints,
                          mlir::Value replacement, mlir::Region &owningRegion);

private:
  fir::AllocaRewriterCallBack allocaRewriter;
  fir::DeallocCallBack deallocGenerator;
  mlir::DominanceInfo dominanceInfo;
};
} // namespace

static bool terminatorYieldsMemory(mlir::Operation &terminator) {
  return llvm::any_of(terminator.getResults(), [](mlir::OpResult res) {
    return fir::conformsWithPassByRef(res.getType());
  });
}

static bool isRegionTerminator(mlir::Operation &terminator) {
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

mlir::Region *AllocaReplaceImpl::findDeallocationPointsAndOwner(
    fir::AllocaOp alloca,
    llvm::SmallVectorImpl<mlir::Operation *> &deallocationPoints) {
  // Step 1: Identify the operation and region owning the alloca.
  mlir::Region *owningRegion = alloca.getOwnerRegion();
  if (!owningRegion)
    return nullptr;
  mlir::Operation *owningOp = owningRegion->getParentOp();
  assert(owningOp && "region expected to be owned");
  // Step 2: Identify the exit points of the owning region, they are the default
  // deallocation points. TODO: detect and use lifetime markers to get earlier
  // deallocation points.
  bool isOpenACCMPRecipe = mlir::isa<mlir::accomp::RecipeInterface>(owningOp);
  for (mlir::Block &block : owningRegion->getBlocks())
    if (mlir::Operation *terminator = block.getTerminator();
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
  // If the owningRegion did not adhere to the ReturnLike interface for its
  // terminators, bail and do not attempt to translate it (we could maybe
  // fallback to consider terminators with no block successor, but since all
  // FIR, OpenACC, OpenMP, CUF, SCF operations with IsIsolatedFromAbove,
  // AutomaticAllocationScope, or LoopLikeOpInterface have such terminators,
  // avoid any untested complexity for now).
  if (deallocationPoints.empty())
    return nullptr;

  // Step 3: detect loops between the alloc and deallocation points.
  // If such loop exists, the easy solution is to consider the alloc
  // as a deallocation point of any previous allocation. This works
  // because the alloc does not properly dominates itself, so the
  // inserted deallocation will be conditional.
  // For now, always assume there may always be a loop if any of the
  // deallocation point does not dominate the alloca. It is
  // conservative approach. Bringing lifetime markers above will reduce
  // the false positive for alloca made inside if like constructs or CFG.
  if (!allocDominatesDealloc(alloca, deallocationPoints))
    deallocationPoints.push_back(alloca.getOperation());
  return owningRegion;
}

static mlir::Value castIfNeeed(mlir::Location loc, mlir::RewriterBase &rewriter,
                               mlir::Type newType, mlir::Value value) {
  if (value.getType() != newType)
    return rewriter.create<fir::ConvertOp>(loc, newType, value);
  return value;
}

void AllocaReplaceImpl::genIndirectDeallocation(
    mlir::RewriterBase &rewriter, fir::AllocaOp alloca,
    llvm::ArrayRef<mlir::Operation *> deallocationPoints,
    mlir::Value replacement, mlir::Region &owningRegion) {
  mlir::Location loc = alloca.getLoc();
  auto replacementInsertPoint = rewriter.saveInsertionPoint();
  // Create C pointer variable in the entry block to store the alloc
  // and access it indirectly in the entry points that do not dominate.
  rewriter.setInsertionPointToStart(&owningRegion.front());
  mlir::Type heapType = fir::HeapType::get(alloca.getInType());
  mlir::Value ptrVar = rewriter.create<fir::AllocaOp>(loc, heapType);
  mlir::Value nullPtr = rewriter.create<fir::ZeroOp>(loc, heapType);
  rewriter.create<fir::StoreOp>(loc, nullPtr, ptrVar);
  // TODO: introducing a pointer compare op in FIR would help
  // generating less IR here.
  mlir::Type intPtrTy = rewriter.getI64Type();
  mlir::Value c0 = rewriter.create<mlir::arith::ConstantOp>(
      loc, intPtrTy, rewriter.getIntegerAttr(intPtrTy, 0));

  // Store new storage address right after its creation.
  rewriter.restoreInsertionPoint(replacementInsertPoint);
  mlir::Value castReplacement =
      castIfNeeed(loc, rewriter, heapType, replacement);
  rewriter.create<fir::StoreOp>(loc, castReplacement, ptrVar);

  // Generate conditional deallocation at every deallocation point.
  auto genConditionalDealloc = [&](mlir::Location loc) {
    mlir::Value ptrVal = rewriter.create<fir::LoadOp>(loc, ptrVar);
    mlir::Value ptrToInt =
        rewriter.create<fir::ConvertOp>(loc, intPtrTy, ptrVal);
    mlir::Value isAllocated = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, ptrToInt, c0);
    auto ifOp = rewriter.create<fir::IfOp>(loc, std::nullopt, isAllocated,
                                           /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    mlir::Value cast =
        castIfNeeed(loc, rewriter, replacement.getType(), ptrVal);
    deallocGenerator(loc, rewriter, cast);
    // Currently there is no need to reset the pointer var because two
    // deallocation points can never be reached without going through the
    // alloca.
    rewriter.setInsertionPointAfter(ifOp);
  };
  for (mlir::Operation *deallocPoint : deallocationPoints) {
    rewriter.setInsertionPoint(deallocPoint);
    genConditionalDealloc(deallocPoint->getLoc());
  }
}

bool AllocaReplaceImpl::replace(mlir::RewriterBase &rewriter,
                                fir::AllocaOp alloca) {
  llvm::SmallVector<mlir::Operation *> deallocationPoints;
  mlir::Region *owningRegion =
      findDeallocationPointsAndOwner(alloca, deallocationPoints);
  if (!owningRegion)
    return false;
  // return false;
  rewriter.setInsertionPointAfter(alloca.getOperation());
  bool deallocPointsDominateAlloc =
      allocDominatesDealloc(alloca, deallocationPoints);
  if (mlir::Value replacement =
          allocaRewriter(rewriter, alloca, deallocPointsDominateAlloc)) {
    mlir::Value castReplacement =
        castIfNeeed(alloca.getLoc(), rewriter, alloca.getType(), replacement);
    if (deallocPointsDominateAlloc)
      for (mlir::Operation *deallocPoint : deallocationPoints) {
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

bool fir::replaceAllocas(mlir::RewriterBase &rewriter,
                         mlir::Operation *parentOp,
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
