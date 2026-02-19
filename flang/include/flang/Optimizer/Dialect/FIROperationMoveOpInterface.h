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
} // namespace fir::detail

#include "flang/Optimizer/Dialect/FIROperationMoveOpInterface.h.inc"

namespace fir {
/// Returns true if it is allowed to move the given 'candidate'
/// operation from the 'descendant' operation into operation 'op'.
/// If 'candidate' is nullptr, then the caller is querying whether
/// any operation from any descendant can be moved into this operation.
bool canMoveFromDescendant(mlir::Operation *op, mlir::Operation *descendant,
                           mlir::Operation *candidate);

/// Returns true if it is allowed to move the given 'candidate'
/// operation out of operation 'op'. If 'candidate' is nullptr,
/// then the caller is querying whether any operation can be moved
/// out of this operation.
bool canMoveOutOf(mlir::Operation *op, mlir::Operation *candidate);
} // namespace fir

#endif // FORTRAN_OPTIMIZER_DIALECT_FIR_OPERATION_MOVE_OP_INTERFACE_H
