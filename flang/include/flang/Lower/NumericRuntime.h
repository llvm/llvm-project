//===-- Lower/NumericRuntime.h -- lower numeric intrinsics --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_NUMERICRUNTIME_H
#define FORTRAN_LOWER_NUMERICRUNTIME_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace fir {
class ExtendedValue;
}

namespace Fortran::lower {
class FirOpBuilder;

/// Generate call to RRSpacing intrinsic runtime routine.
mlir::Value
genRRSpacing(Fortran::lower::FirOpBuilder &builder,
           mlir::Location loc, mlir::Value x);

/// Generate call to Spacing intrinsic runtime routine.
mlir::Value
genSpacing(Fortran::lower::FirOpBuilder &builder,
           mlir::Location loc, mlir::Value x);

}
#endif
