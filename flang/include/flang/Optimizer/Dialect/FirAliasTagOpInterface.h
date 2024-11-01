//===- FirAliasTagOpInterface.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an interface for adding alias analysis information to
// loads and stores
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_FIR_ALIAS_TAG_OP_INTERFACE_H
#define FORTRAN_OPTIMIZER_DIALECT_FIR_ALIAS_TAG_OP_INTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace fir::detail {
mlir::LogicalResult verifyFirAliasTagOpInterface(mlir::Operation *op);
} // namespace fir::detail

#include "flang/Optimizer/Dialect/FirAliasTagOpInterface.h.inc"

#endif // FORTRAN_OPTIMIZER_DIALECT_FIR_ALIAS_TAG_OP_INTERFACE_H
