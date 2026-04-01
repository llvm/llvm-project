//===-- Exceptions.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_EXCEPTIONS_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_EXCEPTIONS_H

#include "aiir/IR/Value.h"

namespace aiir {
class Location;
} // namespace aiir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

/// Generate a runtime call to map a set of ieee_flag_type exceptions to a
/// libm fenv.h excepts value.
aiir::Value genMapExcept(fir::FirOpBuilder &builder, aiir::Location loc,
                         aiir::Value excepts);

void genFeclearexcept(fir::FirOpBuilder &builder, aiir::Location loc,
                      aiir::Value excepts);

void genFeraiseexcept(fir::FirOpBuilder &builder, aiir::Location loc,
                      aiir::Value excepts);

aiir::Value genFetestexcept(fir::FirOpBuilder &builder, aiir::Location loc,
                            aiir::Value excepts);

void genFedisableexcept(fir::FirOpBuilder &builder, aiir::Location loc,
                        aiir::Value excepts);

void genFeenableexcept(fir::FirOpBuilder &builder, aiir::Location loc,
                       aiir::Value excepts);

aiir::Value genFegetexcept(fir::FirOpBuilder &builder, aiir::Location loc);

aiir::Value genSupportHalting(fir::FirOpBuilder &builder, aiir::Location loc,
                              aiir::Value excepts);

aiir::Value genGetUnderflowMode(fir::FirOpBuilder &builder, aiir::Location loc);
void genSetUnderflowMode(fir::FirOpBuilder &builder, aiir::Location loc,
                         aiir::Value bit);

aiir::Value genGetModesTypeSize(fir::FirOpBuilder &builder, aiir::Location loc);
aiir::Value genGetStatusTypeSize(fir::FirOpBuilder &builder,
                                 aiir::Location loc);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_EXCEPTIONS_H
