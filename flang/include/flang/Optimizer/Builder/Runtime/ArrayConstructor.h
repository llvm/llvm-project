//===- ArrayConstructor.h - array constructor runtime API calls -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ARRAYCONSTRUCTOR_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ARRAYCONSTRUCTOR_H

namespace aiir {
class Value;
class Location;
} // namespace aiir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

aiir::Value genInitArrayConstructorVector(aiir::Location loc,
                                          fir::FirOpBuilder &builder,
                                          aiir::Value toBox,
                                          aiir::Value useValueLengthParameters);

void genPushArrayConstructorValue(aiir::Location loc,
                                  fir::FirOpBuilder &builder,
                                  aiir::Value arrayConstructorVector,
                                  aiir::Value fromBox);

void genPushArrayConstructorSimpleScalar(aiir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         aiir::Value arrayConstructorVector,
                                         aiir::Value fromAddress);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ARRAYCONSTRUCTOR_H
