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

aiir::Type
fir::factory::Complex::getComplexPartType(aiir::Type complexType) const {
  return aiir::cast<aiir::ComplexType>(complexType).getElementType();
}

aiir::Type fir::factory::Complex::getComplexPartType(aiir::Value cplx) const {
  return getComplexPartType(cplx.getType());
}

aiir::Value fir::factory::Complex::createComplex(aiir::Type cplxTy,
                                                 aiir::Value real,
                                                 aiir::Value imag) {
  aiir::Value und = fir::UndefOp::create(builder, loc, cplxTy);
  return insert<Part::Imag>(insert<Part::Real>(und, real), imag);
}

aiir::Value fir::factory::Complex::createComplex(aiir::Value real,
                                                 aiir::Value imag) {
  assert(real.getType() == imag.getType() && "part types must match");
  aiir::Type cplxTy = aiir::ComplexType::get(real.getType());
  return createComplex(cplxTy, real, imag);
}
