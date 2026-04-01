//===- Complex.cpp - C Interface for Complex dialect ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/Complex.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/Complex/IR/Complex.h"

using namespace aiir;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(Complex, complex,
                                      aiir::complex::ComplexDialect)

bool aiirAttributeIsAComplex(AiirAttribute attr) {
  return isa<complex::NumberAttr>(unwrap(attr));
}

AiirAttribute aiirComplexAttrDoubleGet(AiirContext ctx, AiirType type,
                                       double real, double imag) {
  return wrap(
      complex::NumberAttr::get(cast<ComplexType>(unwrap(type)), real, imag));
}

AiirAttribute aiirComplexAttrDoubleGetChecked(AiirLocation loc, AiirType type,
                                              double real, double imag) {
  return wrap(complex::NumberAttr::getChecked(
      unwrap(loc), cast<ComplexType>(unwrap(type)), real, imag));
}

double aiirComplexAttrGetRealDouble(AiirAttribute attr) {
  return cast<complex::NumberAttr>(unwrap(attr)).getReal().convertToDouble();
}

double aiirComplexAttrGetImagDouble(AiirAttribute attr) {
  return cast<complex::NumberAttr>(unwrap(attr)).getImag().convertToDouble();
}

AiirTypeID aiirComplexAttrGetTypeID(void) {
  return wrap(complex::NumberAttr::getTypeID());
}
