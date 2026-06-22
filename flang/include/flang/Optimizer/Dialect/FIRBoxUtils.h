//===-- Optimizer/Dialect/FIRBoxUtils.h -- FIR box utilities --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_FIRBOXUTILS_H
#define FORTRAN_OPTIMIZER_DIALECT_FIRBOXUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {

/// Given \p box of type fir::BaseBoxType representing an array, generate code
/// to fetch the lower bounds, extents, and/or strides from the box. Non-null
/// output pointers receive the corresponding values.
void genDimInfoFromBox(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value box,
                       llvm::SmallVectorImpl<mlir::Value> *lbounds,
                       llvm::SmallVectorImpl<mlir::Value> *extents,
                       llvm::SmallVectorImpl<mlir::Value> *strides);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_DIALECT_FIRBOXUTILS_H
