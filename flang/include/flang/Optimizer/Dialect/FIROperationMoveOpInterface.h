//===- FIROperationMoveOpInterface.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares methods used by OperationMoveOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_FIR_OPERATION_MOVE_OP_INTERFACE_H
#define FORTRAN_OPTIMIZER_DIALECT_FIR_OPERATION_MOVE_OP_INTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/LogicalResult.h"

namespace fir::detail {
/// Verify invariants of OperationMoveOpInterface.
llvm::LogicalResult verifyOperationMoveOpInterface(mlir::Operation *op);

/// A wrapper around canMoveFromDescendantImpl().
/// The wrapper asserts certain assumptions about the passed
/// arguments.
bool canMoveFromDescendant(mlir::Operation *op, mlir::Operation *descendant,
                           mlir::Operation *candidate);

/// A wrapper around canMoveOutOfImpl().
/// The wrapper asserts certain assumptions about the passed
/// arguments.
bool canMoveOutOf(mlir::Operation *op, mlir::Operation *candidate);
} // namespace fir::detail

#include "flang/Optimizer/Dialect/FIROperationMoveOpInterface.h.inc"

#endif // FORTRAN_OPTIMIZER_DIALECT_FIR_OPERATION_MOVE_OP_INTERFACE_H
