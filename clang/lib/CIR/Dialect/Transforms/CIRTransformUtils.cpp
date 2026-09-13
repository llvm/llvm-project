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

  // Preserve the call semantics shared by CallOp and TryCallOp. The callee and
  // operand segments are already populated by TryCallOp::create, and a
  // throwing call cannot carry the nothrow property.
  tryCallOp.setInlineKind(callOp.getInlineKind());
  tryCallOp.setMusttail(callOp.getMusttail());
  tryCallOp.setSideEffect(callOp.getSideEffect());
  if (mlir::ArrayAttr argAttrs = callOp.getArgAttrsAttr())
    tryCallOp.setArgAttrsAttr(argAttrs);
  if (mlir::ArrayAttr resAttrs = callOp.getResAttrsAttr())
    tryCallOp.setResAttrsAttr(resAttrs);

  for (mlir::NamedAttribute attr : callOp->getDiscardableAttrs()) {
    assert(attr.getName() != cir::CIRDialect::getNoUnwindAttrName() &&
           "unexpected attribute on converted call");
    tryCallOp->setDiscardableAttr(attr.getName(), attr.getValue());
  }

  // Replace uses of the call result with the try_call result. Use the
  // rewriter API so any listener (e.g. the pattern rewriter in
  // FlattenCFG) is notified of the in-place modifications to each user.
  if (callOp->getNumResults() > 0)
    rewriter.replaceAllUsesWith(callOp->getResult(0), tryCallOp.getResult());

  rewriter.eraseOp(callOp);
  return normalDest;
}

mlir::Block *cir::replaceThrowWithTryThrow(cir::ThrowOp throwOp,
                                           mlir::Block *unwindDest,
                                           mlir::Location loc,
                                           mlir::RewriterBase &rewriter) {
  // The throw never returns, so the try_throw's normal destination is
  // literally unreachable. Place it at the end of the parent function
  // rather than splitting it out of the throw's block in the middle of
  // the normal control flow.
  auto funcOp = throwOp->getParentOfType<cir::FuncOp>();
  assert(funcOp && "throw must be inside a function");
  mlir::Region &body = funcOp.getBody();

  mlir::Block *normalDest;
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    normalDest = rewriter.createBlock(&body, body.end());
    cir::UnreachableOp::create(rewriter, loc);
  }

  // Build the try_throw to replace the original throw.
  rewriter.setInsertionPoint(throwOp);
  auto tryThrowOp = cir::TryThrowOp::create(
      rewriter, loc, throwOp.getExceptionPtr(), throwOp.getTypeInfoAttr(),
      throwOp.getDtorAttr(), normalDest, unwindDest);

  // The shared inherent state is already set by TryThrowOp::create. Preserve
  // only auxiliary metadata here.
  for (mlir::NamedAttribute attr : throwOp->getDiscardableAttrs())
    tryThrowOp->setDiscardableAttr(attr.getName(), attr.getValue());

  // Erase the throw along with any operations that followed it in its
  // parent block (typically a cir.unreachable left over from CIR codegen).
  // They must be removed because try_throw is a terminator and a block
  // can have only one terminator.
  mlir::Block *throwBlock = throwOp->getBlock();
  while (&throwBlock->back() != tryThrowOp)
    rewriter.eraseOp(&throwBlock->back());

  return normalDest;
}
