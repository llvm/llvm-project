//===- CIRTransformUtils.cpp - Shared helpers for CIR transforms ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRTransformUtils.h"

#include "clang/CIR/Dialect/IR/CIRTypes.h"

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
