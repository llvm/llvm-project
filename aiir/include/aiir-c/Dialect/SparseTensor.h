//===-- aiir-c/Dialect/SparseTensor.h - C API for SparseTensor ----*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_SPARSETENSOR_H
#define AIIR_C_DIALECT_SPARSETENSOR_H

#include "aiir-c/AffineMap.h"
#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(SparseTensor, sparse_tensor);

/// Dimension level types (and properties) that define sparse tensors.
/// See the documentation in SparseTensorAttrDefs.td for their meaning.
///
/// These correspond to SparseTensorEncodingAttr::LevelType in the C++ API.
/// If updating, keep them in sync and update the static_assert in the impl
/// file.
typedef uint64_t AiirSparseTensorLevelType;

enum AiirSparseTensorLevelFormat {
  AIIR_SPARSE_TENSOR_LEVEL_DENSE = 0x000000010000,
  AIIR_SPARSE_TENSOR_LEVEL_BATCH = 0x000000020000,
  AIIR_SPARSE_TENSOR_LEVEL_COMPRESSED = 0x000000040000,
  AIIR_SPARSE_TENSOR_LEVEL_SINGLETON = 0x000000080000,
  AIIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED = 0x000000100000,
  AIIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M = 0x000000200000,
};

enum AiirSparseTensorLevelPropertyNondefault {
  AIIR_SPARSE_PROPERTY_NON_UNIQUE = 0x0001,
  AIIR_SPARSE_PROPERTY_NON_ORDERED = 0x0002,
  AIIR_SPARSE_PROPERTY_SOA = 0x0004,
};

//===----------------------------------------------------------------------===//
// SparseTensorEncodingAttr
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a `sparse_tensor.encoding` attribute.
AIIR_CAPI_EXPORTED bool
aiirAttributeIsASparseTensorEncodingAttr(AiirAttribute attr);

/// Creates a `sparse_tensor.encoding` attribute with the given parameters.
AIIR_CAPI_EXPORTED AiirAttribute aiirSparseTensorEncodingAttrGet(
    AiirContext ctx, intptr_t lvlRank,
    AiirSparseTensorLevelType const *lvlTypes, AiirAffineMap dimToLvl,
    AiirAffineMap lvlTodim, int posWidth, int crdWidth,
    AiirAttribute explicitVal, AiirAttribute implicitVal);

AIIR_CAPI_EXPORTED AiirStringRef aiirSparseTensorEncodingAttrGetName(void);

/// Returns the level-rank of the `sparse_tensor.encoding` attribute.
AIIR_CAPI_EXPORTED intptr_t
aiirSparseTensorEncodingGetLvlRank(AiirAttribute attr);

/// Returns a specified level-type of the `sparse_tensor.encoding` attribute.
AIIR_CAPI_EXPORTED AiirSparseTensorLevelType
aiirSparseTensorEncodingAttrGetLvlType(AiirAttribute attr, intptr_t lvl);

/// Returns a specified level-format of the `sparse_tensor.encoding` attribute.
AIIR_CAPI_EXPORTED enum AiirSparseTensorLevelFormat
aiirSparseTensorEncodingAttrGetLvlFmt(AiirAttribute attr, intptr_t lvl);

/// Returns the dimension-to-level mapping of the `sparse_tensor.encoding`
/// attribute.
AIIR_CAPI_EXPORTED AiirAffineMap
aiirSparseTensorEncodingAttrGetDimToLvl(AiirAttribute attr);

/// Returns the level-to-dimension mapping of the `sparse_tensor.encoding`
/// attribute.
AIIR_CAPI_EXPORTED AiirAffineMap
aiirSparseTensorEncodingAttrGetLvlToDim(AiirAttribute attr);

/// Returns the position bitwidth of the `sparse_tensor.encoding` attribute.
AIIR_CAPI_EXPORTED int
aiirSparseTensorEncodingAttrGetPosWidth(AiirAttribute attr);

/// Returns the coordinate bitwidth of the `sparse_tensor.encoding` attribute.
AIIR_CAPI_EXPORTED int
aiirSparseTensorEncodingAttrGetCrdWidth(AiirAttribute attr);

/// Returns the explicit value of the `sparse_tensor.encoding` attribute.
AIIR_CAPI_EXPORTED AiirAttribute
aiirSparseTensorEncodingAttrGetExplicitVal(AiirAttribute attr);

/// Returns the implicit value of the `sparse_tensor.encoding` attribute.
AIIR_CAPI_EXPORTED AiirAttribute
aiirSparseTensorEncodingAttrGetImplicitVal(AiirAttribute attr);

AIIR_CAPI_EXPORTED unsigned
aiirSparseTensorEncodingAttrGetStructuredN(AiirSparseTensorLevelType lvlType);

AIIR_CAPI_EXPORTED unsigned
aiirSparseTensorEncodingAttrGetStructuredM(AiirSparseTensorLevelType lvlType);

AIIR_CAPI_EXPORTED AiirSparseTensorLevelType
aiirSparseTensorEncodingAttrBuildLvlType(
    enum AiirSparseTensorLevelFormat lvlFmt,
    const enum AiirSparseTensorLevelPropertyNondefault *properties,
    unsigned propSize, unsigned n, unsigned m);

#ifdef __cplusplus
}
#endif

#include "aiir/Dialect/SparseTensor/Transforms/Passes.capi.h.inc"

#endif // AIIR_C_DIALECT_SPARSETENSOR_H
