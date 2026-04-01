//===-- aiir-c/Dialect/Complex.h - C API for Complex dialect ------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_COMPLEX_H
#define AIIR_C_DIALECT_COMPLEX_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(Complex, complex);

/// Checks whether the given attribute is a complex attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsAComplex(AiirAttribute attr);

/// Creates a complex attribute in the given context with the given
/// double real and imaginary values and double-precision FP semantics.
AIIR_CAPI_EXPORTED AiirAttribute aiirComplexAttrDoubleGet(AiirContext ctx,
                                                          AiirType type,
                                                          double real,
                                                          double imag);

/// Same as "aiirComplexAttrDoubleGet", but if the type is not valid for a
/// construction of a ComplexAttr, returns a null AiirAttribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirComplexAttrDoubleGetChecked(
    AiirLocation loc, AiirType type, double real, double imag);

/// Returns the real value stored in the given complex attribute, interpreting
/// the value as double.
AIIR_CAPI_EXPORTED double aiirComplexAttrGetRealDouble(AiirAttribute attr);

/// Returns the imaginaryvalue stored in the given complex attribute,
/// interpreting the value as double.
AIIR_CAPI_EXPORTED double aiirComplexAttrGetImagDouble(AiirAttribute attr);

/// Returns the typeID of a Complex attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirComplexAttrGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_DIALECT_COMPLEX_H
