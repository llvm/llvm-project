//===- CIRTransformUtils.cpp - Shared helpers for CIR transforms ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/CIRTransformUtils.h"

#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "llvm/ADT/DepthFirstIterator.h"

void cir::collectUnreachable(mlir::Operation *parent,
                             llvm::SmallVectorImpl<mlir::Operation *> &ops) {
  // For every region under `parent`, find the blocks unreachable from the
  // entry via a forward CFG traversal and collect their ops.
  llvm::df_iterator_default_set<mlir::Block *, 16> reachable;
  parent->walk([&](mlir::Region *region) {
    // Empty regions have no blocks; single-block regions have only the
    // entry, which is trivially reachable. Either way, nothing to collect.
    if (region->empty() || region->hasOneBlock())
      return;

    // We clear this for each region as we walk the parent because each block
    // is only in one region, so the reachable blocks from previously visited
    // regions aren't needed.
    reachable.clear();

    // The depth_first_ext range iterator internally adds each block to the
    // reachable set as it visits it, so while this loop looks like it doesn't
    // do anything, it's actually populating the set of reachable blocks in
    // this region.
    for (mlir::Block *blk : llvm::depth_first_ext(&region->front(), reachable))
      (void)blk;

    // Collect the unreachable blocks.
    for (mlir::Block &blk : *region) {
      if (reachable.contains(&blk))
        continue;
      for (mlir::Operation &op : blk)
        ops.push_back(&op);
    }
  });
}

mlir::Block *cir::replaceCallWithTryCall(cir::CallOp callOp,
                                         mlir::Block *unwindDest,
                                         mlir::Location loc,
                                         mlir::RewriterBase &rewriter) {
  mlir::Block *callBlock = callOp->getBlock();

  assert(!callOp.getNothrow() && "call is not expected to throw");

  // Split the block after the call - remaining ops become the normal
  // destination.
  mlir::Block *normalDest =
      rewriter.splitBlock(callBlock, std::next(callOp->getIterator()));

  // Build the try_call to replace the original call.
  rewriter.setInsertionPoint(callOp);
  cir::TryCallOp tryCallOp;
  if (callOp.isIndirect()) {
    mlir::Value indTarget = callOp.getIndirectCall();
    auto ptrTy = mlir::cast<cir::PointerType>(indTarget.getType());
    auto resTy = mlir::cast<cir::FuncType>(ptrTy.getPointee());
    tryCallOp =
        cir::TryCallOp::create(rewriter, loc, indTarget, resTy, normalDest,
                               unwindDest, callOp.getArgOperands());
  } else {
    mlir::Type resType = callOp->getNumResults() > 0
                             ? callOp->getResult(0).getType()
                             : mlir::Type();
    tryCallOp =
        cir::TryCallOp::create(rewriter, loc, callOp.getCalleeAttr(), resType,
                               normalDest, unwindDest, callOp.getArgOperands());
  }

  // Copy all attributes from the original call except those already set by
  // TryCallOp::create or that are operation-specific and should not be copied.
  llvm::StringRef excludedAttrs[] = {
      cir::CIRDialect::getCalleeAttrName(), // Set by create()
      cir::CIRDialect::getOperandSegmentSizesAttrName(),
  };
  for (mlir::NamedAttribute attr : callOp->getAttrs()) {
    if (llvm::is_contained(excludedAttrs, attr.getName()))
      continue;
    assert(!llvm::is_contained(
               {
                   cir::CIRDialect::getNoThrowAttrName(),
                   cir::CIRDialect::getNoUnwindAttrName(),
               },
               attr.getName()) &&
           "unexpected attribute on converted call");
    tryCallOp->setAttr(attr.getName(), attr.getValue());
  }

  // Replace uses of the call result with the try_call result. Use the
  // rewriter API so any listener (e.g. the pattern rewriter in
  // FlattenCFG) is notified of the in-place modifications to each user.
  if (callOp->getNumResults() > 0)
    rewriter.replaceAllUsesWith(callOp->getResult(0), tryCallOp.getResult());

  rewriter.eraseOp(callOp);
  return normalDest;
}
