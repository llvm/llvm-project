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

#include "aiir/Dialect/Func/IR/FuncOps.h"

namespace fir {
class ExtendedValue;
class FirOpBuilder;
} // namespace fir

namespace fir::runtime {

void genBesselJn(fir::FirOpBuilder &builder, aiir::Location loc,
                 aiir::Value resultBox, aiir::Value n1, aiir::Value n2,
                 aiir::Value x, aiir::Value bn2, aiir::Value bn2_1);

void genBesselJnX0(fir::FirOpBuilder &builder, aiir::Location loc,
                   aiir::Type xTy, aiir::Value resultBox, aiir::Value n1,
                   aiir::Value n2);

void genBesselYn(fir::FirOpBuilder &builder, aiir::Location loc,
                 aiir::Value resultBox, aiir::Value n1, aiir::Value n2,
                 aiir::Value x, aiir::Value bn1, aiir::Value bn1_1);

void genBesselYnX0(fir::FirOpBuilder &builder, aiir::Location loc,
                   aiir::Type xTy, aiir::Value resultBox, aiir::Value n1,
                   aiir::Value n2);

void genCshift(fir::FirOpBuilder &builder, aiir::Location loc,
               aiir::Value resultBox, aiir::Value arrayBox,
               aiir::Value shiftBox, aiir::Value dimBox);

void genCshiftVector(fir::FirOpBuilder &builder, aiir::Location loc,
                     aiir::Value resultBox, aiir::Value arrayBox,
                     aiir::Value shiftBox);

void genEoshift(fir::FirOpBuilder &builder, aiir::Location loc,
                aiir::Value resultBox, aiir::Value arrayBox,
                aiir::Value shiftBox, aiir::Value boundBox, aiir::Value dimBox);

void genEoshiftVector(fir::FirOpBuilder &builder, aiir::Location loc,
                      aiir::Value resultBox, aiir::Value arrayBox,
                      aiir::Value shiftBox, aiir::Value boundBox);

void genMatmul(fir::FirOpBuilder &builder, aiir::Location loc,
               aiir::Value matrixABox, aiir::Value matrixBBox,
               aiir::Value resultBox);

void genMatmulTranspose(fir::FirOpBuilder &builder, aiir::Location loc,
                        aiir::Value matrixABox, aiir::Value matrixBBox,
                        aiir::Value resultBox);

void genPack(fir::FirOpBuilder &builder, aiir::Location loc,
             aiir::Value resultBox, aiir::Value arrayBox, aiir::Value maskBox,
             aiir::Value vectorBox);

void genShallowCopy(fir::FirOpBuilder &builder, aiir::Location loc,
                    aiir::Value resultBox, aiir::Value arrayBox,
                    bool resultIsAllocated);

void genReshape(fir::FirOpBuilder &builder, aiir::Location loc,
                aiir::Value resultBox, aiir::Value sourceBox,
                aiir::Value shapeBox, aiir::Value padBox, aiir::Value orderBox);

void genSpread(fir::FirOpBuilder &builder, aiir::Location loc,
               aiir::Value resultBox, aiir::Value sourceBox, aiir::Value dim,
               aiir::Value ncopies);

void genTranspose(fir::FirOpBuilder &builder, aiir::Location loc,
                  aiir::Value resultBox, aiir::Value sourceBox);

void genUnpack(fir::FirOpBuilder &builder, aiir::Location loc,
               aiir::Value resultBox, aiir::Value vectorBox,
               aiir::Value maskBox, aiir::Value fieldBox);

} // namespace fir::runtime

#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_TRANSFORMATIONAL_H
