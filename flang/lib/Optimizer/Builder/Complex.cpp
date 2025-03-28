//===-- Complex.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Complex.h"

//===----------------------------------------------------------------------===//
// Complex Factory implementation
//===----------------------------------------------------------------------===//

mlir::Type
fir::factory::Complex::getComplexPartType(mlir::Type complexType) const {
  return mlir::cast<mlir::ComplexType>(complexType).getElementType();
}

mlir::Type fir::factory::Complex::getComplexPartType(mlir::Value cplx) const {
  return getComplexPartType(cplx.getType());
}

mlir::Value fir::factory::Complex::createComplex(mlir::Type cplxTy,
                                                 mlir::Value real,
                                                 mlir::Value imag) {
  mlir::Value und = builder.create<fir::UndefOp>(loc, cplxTy);
  return insert<Part::Imag>(insert<Part::Real>(und, real), imag);
}

mlir::Value fir::factory::Complex::createComplex(mlir::Value real,
                                                 mlir::Value imag) {
  assert(real.getType() == imag.getType() && "part types must match");
  mlir::Type cplxTy = mlir::ComplexType::get(real.getType());
  return createComplex(cplxTy, real, imag);
}
