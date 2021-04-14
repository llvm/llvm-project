//===-- Lower/ReductionRuntime.h -- lower reduction intrinsics --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_REDUCTIONRUNTIME_H
#define FORTRAN_LOWER_REDUCTIONRUNTIME_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace fir {
class ExtendedValue;
}

namespace Fortran::lower {
class FirOpBuilder;

/// Generate call to all runtime routine.
/// This calls the descriptor based runtime call implementation of the scan
/// intrinsics.
void genAllDescriptor(Fortran::lower::FirOpBuilder &builder, 
                         mlir::Location loc,
                         mlir::Value resultBox, mlir::Value maskBox,
                         mlir::Value dim); 

/// Generate call to any runtime routine.
/// This calls the descriptor based runtime call implementation of the scan
/// intrinsics.
void genAnyDescriptor(Fortran::lower::FirOpBuilder &builder, 
                         mlir::Location loc,
                         mlir::Value resultBox, mlir::Value maskBox,
                         mlir::Value dim); 

/// Generate call to all runtime routine. This version of all is specialized
/// for rank 1 mask arguments.
/// This calls the version that returns a scalar logical value.
mlir::Value
genAll(Fortran::lower::FirOpBuilder &builder,
       mlir::Location loc, mlir::Value maskBox,
       mlir::Value dim);

/// Generate call to any runtime routine. This version of any is specialized
/// for rank 1 mask arguments.
/// This calls the version that returns a scalar logical value.
mlir::Value
genAny(Fortran::lower::FirOpBuilder &builder,
       mlir::Location loc, mlir::Value maskBox,
       mlir::Value dim);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_REDUCTIONRUNTIME_H
