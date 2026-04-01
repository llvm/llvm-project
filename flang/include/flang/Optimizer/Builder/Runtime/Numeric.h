//===-- Numeric.h -- generate numeric intrinsics runtime calls --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_NUMERIC_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_NUMERIC_H

#include "aiir/Dialect/Func/IR/FuncOps.h"

namespace fir {
class ExtendedValue;
class FirOpBuilder;
} // namespace fir

namespace fir::runtime {

/// Generate call to ErfcScaled intrinsic runtime routine.
aiir::Value genErfcScaled(fir::FirOpBuilder &builder, aiir::Location loc,
                          aiir::Value x);

/// Generate call to Exponent intrinsic runtime routine.
aiir::Value genExponent(fir::FirOpBuilder &builder, aiir::Location loc,
                        aiir::Type resultType, aiir::Value x);

/// Generate call to Fraction intrinsic runtime routine.
aiir::Value genFraction(fir::FirOpBuilder &builder, aiir::Location loc,
                        aiir::Value x);

/// Generate call to Mod intrinsic runtime routine.
aiir::Value genMod(fir::FirOpBuilder &builder, aiir::Location loc,
                   aiir::Value a, aiir::Value p);

/// Generate call to Modulo intrinsic runtime routine.
aiir::Value genModulo(fir::FirOpBuilder &builder, aiir::Location loc,
                      aiir::Value a, aiir::Value p);

/// Generate call to Nearest intrinsic runtime routine.
aiir::Value genNearest(fir::FirOpBuilder &builder, aiir::Location loc,
                       aiir::Value x, aiir::Value s);

/// Generate call to RRSpacing intrinsic runtime routine.
aiir::Value genRRSpacing(fir::FirOpBuilder &builder, aiir::Location loc,
                         aiir::Value x);

/// Generate call to Scale intrinsic runtime routine.
aiir::Value genScale(fir::FirOpBuilder &builder, aiir::Location loc,
                     aiir::Value x, aiir::Value i);

/// Generate call to Selected_char_kind intrinsic runtime routine.
aiir::Value genSelectedCharKind(fir::FirOpBuilder &builder, aiir::Location loc,
                                aiir::Value name, aiir::Value length);

/// Generate call to Selected_int_kind intrinsic runtime routine.
aiir::Value genSelectedIntKind(fir::FirOpBuilder &builder, aiir::Location loc,
                               aiir::Value x);

/// Generate call to Selected_logical_kind intrinsic runtime routine.
aiir::Value genSelectedLogicalKind(fir::FirOpBuilder &builder,
                                   aiir::Location loc, aiir::Value x);

/// Generate call to Selected_real_kind intrinsic runtime routine.
aiir::Value genSelectedRealKind(fir::FirOpBuilder &builder, aiir::Location loc,
                                aiir::Value precision, aiir::Value range,
                                aiir::Value radix);

/// Generate call to Set_exponent intrinsic runtime routine.
aiir::Value genSetExponent(fir::FirOpBuilder &builder, aiir::Location loc,
                           aiir::Value x, aiir::Value i);

/// Generate call to Spacing intrinsic runtime routine.
aiir::Value genSpacing(fir::FirOpBuilder &builder, aiir::Location loc,
                       aiir::Value x);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_NUMERIC_H
