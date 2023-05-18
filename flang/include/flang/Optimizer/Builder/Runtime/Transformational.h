//===-- Transformational.h --------------------------------------*- C++ -*-===//
// Generate transformational intrinsic runtime API calls.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_TRANSFORMATIONAL_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_TRANSFORMATIONAL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace fir {
class ExtendedValue;
class FirOpBuilder;
} // namespace fir

namespace fir::runtime {

void genBesselJn(fir::FirOpBuilder &builder, mlir::Location loc,
                 mlir::Value resultBox, mlir::Value n1, mlir::Value n2,
                 mlir::Value x, mlir::Value bn2, mlir::Value bn2_1);

void genBesselJnX0(fir::FirOpBuilder &builder, mlir::Location loc,
                   mlir::Type xTy, mlir::Value resultBox, mlir::Value n1,
                   mlir::Value n2);

void genBesselYn(fir::FirOpBuilder &builder, mlir::Location loc,
                 mlir::Value resultBox, mlir::Value n1, mlir::Value n2,
                 mlir::Value x, mlir::Value bn1, mlir::Value bn1_1);

void genBesselYnX0(fir::FirOpBuilder &builder, mlir::Location loc,
                   mlir::Type xTy, mlir::Value resultBox, mlir::Value n1,
                   mlir::Value n2);

void genCshift(fir::FirOpBuilder &builder, mlir::Location loc,
               mlir::Value resultBox, mlir::Value arrayBox,
               mlir::Value shiftBox, mlir::Value dimBox);

void genCshiftVector(fir::FirOpBuilder &builder, mlir::Location loc,
                     mlir::Value resultBox, mlir::Value arrayBox,
                     mlir::Value shiftBox);

void genEoshift(fir::FirOpBuilder &builder, mlir::Location loc,
                mlir::Value resultBox, mlir::Value arrayBox,
                mlir::Value shiftBox, mlir::Value boundBox, mlir::Value dimBox);

void genEoshiftVector(fir::FirOpBuilder &builder, mlir::Location loc,
                      mlir::Value resultBox, mlir::Value arrayBox,
                      mlir::Value shiftBox, mlir::Value boundBox);

void genMatmul(fir::FirOpBuilder &builder, mlir::Location loc,
               mlir::Value matrixABox, mlir::Value matrixBBox,
               mlir::Value resultBox);

void genMatmulTranspose(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::Value matrixABox, mlir::Value matrixBBox,
                        mlir::Value resultBox);

void genPack(fir::FirOpBuilder &builder, mlir::Location loc,
             mlir::Value resultBox, mlir::Value arrayBox, mlir::Value maskBox,
             mlir::Value vectorBox);

void genReshape(fir::FirOpBuilder &builder, mlir::Location loc,
                mlir::Value resultBox, mlir::Value sourceBox,
                mlir::Value shapeBox, mlir::Value padBox, mlir::Value orderBox);

void genSpread(fir::FirOpBuilder &builder, mlir::Location loc,
               mlir::Value resultBox, mlir::Value sourceBox, mlir::Value dim,
               mlir::Value ncopies);

void genTranspose(fir::FirOpBuilder &builder, mlir::Location loc,
                  mlir::Value resultBox, mlir::Value sourceBox);

void genUnpack(fir::FirOpBuilder &builder, mlir::Location loc,
               mlir::Value resultBox, mlir::Value vectorBox,
               mlir::Value maskBox, mlir::Value fieldBox);

} // namespace fir::runtime

#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_TRANSFORMATIONAL_H
