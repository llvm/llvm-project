//===-- Numeric.h -- generate numeric intrinsics runtime calls --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_NUMERIC_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_NUMERIC_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace fir {
class ExtendedValue;
class FirOpBuilder;
} // namespace fir

namespace fir::runtime {

/// Generate call to Exponent intrinsic runtime routine.
mlir::Value genExponent(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::Type resultType, mlir::Value x);

/// Generate call to Fraction intrinsic runtime routine.
mlir::Value genFraction(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::Value x);

/// Generate call to Mod intrinsic runtime routine.
mlir::Value genMod(fir::FirOpBuilder &builder, mlir::Location loc,
                   mlir::Value a, mlir::Value p);

/// Generate call to Modulo intrinsic runtime routine.
mlir::Value genModulo(fir::FirOpBuilder &builder, mlir::Location loc,
                      mlir::Value a, mlir::Value p);

/// Generate call to Nearest intrinsic runtime routine.
mlir::Value genNearest(fir::FirOpBuilder &builder, mlir::Location loc,
                       mlir::Value x, mlir::Value s);

/// Generate call to RRSpacing intrinsic runtime routine.
mlir::Value genRRSpacing(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value x);

/// Generate call to Scale intrinsic runtime routine.
mlir::Value genScale(fir::FirOpBuilder &builder, mlir::Location loc,
                     mlir::Value x, mlir::Value i);

/// Generate call to Selected_int_kind intrinsic runtime routine.
mlir::Value genSelectedIntKind(fir::FirOpBuilder &builder, mlir::Location loc,
                               mlir::Value x);

/// Generate call to Selected_real_kind intrinsic runtime routine.
mlir::Value genSelectedRealKind(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value precision, mlir::Value range,
                                mlir::Value radix);

/// Generate call to Set_exponent intrinsic runtime routine.
mlir::Value genSetExponent(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value x, mlir::Value i);

/// Generate call to Spacing intrinsic runtime routine.
mlir::Value genSpacing(fir::FirOpBuilder &builder, mlir::Location loc,
                       mlir::Value x);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_NUMERIC_H
