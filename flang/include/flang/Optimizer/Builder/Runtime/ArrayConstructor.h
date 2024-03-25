//===- ArrayConstructor.h - array constructor runtime API calls -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ARRAYCONSTRUCTOR_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ARRAYCONSTRUCTOR_H

namespace mlir {
class Value;
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

mlir::Value genInitArrayConstructorVector(mlir::Location loc,
                                          fir::FirOpBuilder &builder,
                                          mlir::Value toBox,
                                          mlir::Value useValueLengthParameters);

void genPushArrayConstructorValue(mlir::Location loc,
                                  fir::FirOpBuilder &builder,
                                  mlir::Value arrayConstructorVector,
                                  mlir::Value fromBox);

void genPushArrayConstructorSimpleScalar(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         mlir::Value arrayConstructorVector,
                                         mlir::Value fromAddress);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ARRAYCONSTRUCTOR_H
