//===-- aiir-c/Dialect/EmitC.h - C API for EmitC dialect ----------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_EmitC_H
#define AIIR_C_DIALECT_EmitC_H

#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(EmitC, emitc);

enum AiirEmitCCmpPredicate : uint64_t {
  AIIR_EMITC_CMP_PREDICATE_EQ = 0,
  AIIR_EMITC_CMP_PREDICATE_NE = 1,
  AIIR_EMITC_CMP_PREDICATE_LT = 2,
  AIIR_EMITC_CMP_PREDICATE_LE = 3,
  AIIR_EMITC_CMP_PREDICATE_GT = 4,
  AIIR_EMITC_CMP_PREDICATE_GE = 5,
  AIIR_EMITC_CMP_PREDICATE_THREE_WAY = 6,
};

//===---------------------------------------------------------------------===//
// ArrayType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAEmitCArrayType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirEmitCArrayTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirEmitCArrayTypeGet(intptr_t nDims,
                                                  int64_t *shape,
                                                  AiirType elementType);

AIIR_CAPI_EXPORTED AiirStringRef aiirEmitCArrayTypeGetName(void);

//===---------------------------------------------------------------------===//
// LValueType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAEmitCLValueType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirEmitCLValueTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirEmitCLValueTypeGet(AiirType valueType);

AIIR_CAPI_EXPORTED AiirStringRef aiirEmitCLValueTypeGetName(void);

//===---------------------------------------------------------------------===//
// OpaqueType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAEmitCOpaqueType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirEmitCOpaqueTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirEmitCOpaqueTypeGet(AiirContext ctx,
                                                   AiirStringRef value);

AIIR_CAPI_EXPORTED AiirStringRef aiirEmitCOpaqueTypeGetName(void);

//===---------------------------------------------------------------------===//
// PointerType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAEmitCPointerType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirEmitCPointerTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirEmitCPointerTypeGet(AiirType pointee);

AIIR_CAPI_EXPORTED AiirStringRef aiirEmitCPointerTypeGetName(void);

//===---------------------------------------------------------------------===//
// PtrDiffTType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAEmitCPtrDiffTType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirEmitCPtrDiffTTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirEmitCPtrDiffTTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirEmitCPtrDiffTTypeGetName(void);

//===---------------------------------------------------------------------===//
// SignedSizeTType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAEmitCSignedSizeTType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirEmitCSignedSizeTTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirEmitCSignedSizeTTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirEmitCSignedSizeTTypeGetName(void);

//===---------------------------------------------------------------------===//
// SizeTType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAEmitCSizeTType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirEmitCSizeTTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirEmitCSizeTTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirEmitCSizeTTypeGetName(void);

//===----------------------------------------------------------------------===//
// CmpPredicate attribute.
//===----------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirAttributeIsAEmitCCmpPredicate(AiirAttribute attr);

AIIR_CAPI_EXPORTED AiirAttribute
aiirEmitCCmpPredicateAttrGet(AiirContext ctx, enum AiirEmitCCmpPredicate val);

AIIR_CAPI_EXPORTED AiirStringRef aiirEmitCCmpPredicateAttrGetName(void);

AIIR_CAPI_EXPORTED enum AiirEmitCCmpPredicate
aiirEmitCCmpPredicateAttrGetValue(AiirAttribute attr);

AIIR_CAPI_EXPORTED AiirTypeID aiirEmitCCmpPredicateAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Opaque attribute.
//===----------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirAttributeIsAEmitCOpaque(AiirAttribute attr);

AIIR_CAPI_EXPORTED AiirAttribute aiirEmitCOpaqueAttrGet(AiirContext ctx,
                                                        AiirStringRef value);

AIIR_CAPI_EXPORTED AiirStringRef aiirEmitCOpaqueAttrGetName(void);

AIIR_CAPI_EXPORTED AiirStringRef
aiirEmitCOpaqueAttrGetValue(AiirAttribute attr);

AIIR_CAPI_EXPORTED AiirTypeID aiirEmitCOpaqueAttrGetTypeID(void);

#ifdef __cplusplus
}
#endif

#include "aiir/Dialect/EmitC/Transforms/Passes.capi.h.inc"

#endif // AIIR_C_DIALECT_EmitC_H
