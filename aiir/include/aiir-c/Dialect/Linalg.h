//===-- aiir-c/Dialect/Linalg.h - C API for Linalg dialect -------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_LINALG_H
#define AIIR_C_DIALECT_LINALG_H

#include "aiir-c/AffineMap.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Apply the special region builder for the builtin named Linalg op.
/// Assert that `aiirOp` is a builtin named Linalg op.
AIIR_CAPI_EXPORTED void
aiirLinalgFillBuiltinNamedOpRegion(AiirOperation aiirOp);

AIIR_CAPI_EXPORTED bool aiirLinalgIsAContractionOp(AiirOperation op);

typedef struct AiirLinalgContractionDimensions {
  AiirAttribute batch;
  AiirAttribute m;
  AiirAttribute n;
  AiirAttribute k;
} AiirLinalgContractionDimensions;

AIIR_CAPI_EXPORTED AiirLinalgContractionDimensions
aiirLinalgInferContractionDimensions(AiirOperation op);

AIIR_CAPI_EXPORTED AiirLinalgContractionDimensions
aiirLinalgInferContractionDimensionsFromMaps(const AiirAffineMap *indexingMaps,
                                             size_t numMaps);

AIIR_CAPI_EXPORTED bool aiirLinalgIsAConvolutionOp(AiirOperation op);

typedef struct AiirLinalgConvolutionDimensions {
  AiirAttribute batch;
  AiirAttribute outputImage;
  AiirAttribute outputChannel;
  AiirAttribute filterLoop;
  AiirAttribute inputChannel;
  AiirAttribute depth;
  AiirAttribute strides;
  AiirAttribute dilations;
} AiirLinalgConvolutionDimensions;

AIIR_CAPI_EXPORTED AiirLinalgConvolutionDimensions
aiirLinalgInferConvolutionDimensions(AiirOperation op);

AIIR_CAPI_EXPORTED AiirAttribute
aiirLinalgGetIndexingMapsAttribute(AiirOperation op);

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(Linalg, linalg);

#ifdef __cplusplus
}
#endif

#include "aiir/Dialect/Linalg/Passes.capi.h.inc"

#endif // AIIR_C_DIALECT_LINALG_H
