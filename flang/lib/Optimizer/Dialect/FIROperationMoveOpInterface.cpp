//===-- FIROperationMoveOpInterface.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROperationMoveOpInterface.h"

#include "flang/Optimizer/Dialect/FIROperationMoveOpInterface.cpp.inc"

llvm::LogicalResult
fir::detail::verifyOperationMoveOpInterface(mlir::Operation *op) {
  // It does not make sense to use this interface for operations
  // without any regions.
  if (op->getNumRegions() == 0)
    return op->emitOpError("must contain at least one region");
  return llvm::success();
}

bool fir::canMoveFromDescendant(mlir::Operation *op,
                                mlir::Operation *descendant,
                                mlir::Operation *candidate) {
  // Perform some sanity checks.
  assert(op->isProperAncestor(descendant) &&
         "op must be an ancestor of descendant");
  if (candidate)
    assert(descendant->isProperAncestor(candidate) &&
           "descendant must be an ancestor of candidate");
  if (auto iface = mlir::dyn_cast<OperationMoveOpInterface>(op))
    return iface.canMoveFromDescendant(descendant, candidate);

  return true;
}

bool fir::canMoveOutOf(mlir::Operation *op, mlir::Operation *candidate) {
  if (candidate)
    assert(op->isProperAncestor(candidate) &&
           "op must be an ancestor of candidate");
  if (auto iface = mlir::dyn_cast<OperationMoveOpInterface>(op))
    return iface.canMoveOutOf(candidate);

  return true;
}
