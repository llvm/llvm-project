//===- SMT.h - C interface for the SMT dialect --------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_SMT_H
#define AIIR_C_DIALECT_SMT_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(SMT, smt);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Checks if the given type is any non-func SMT value type.
AIIR_CAPI_EXPORTED bool aiirSMTTypeIsAnyNonFuncSMTValueType(AiirType type);

/// Checks if the given type is any SMT value type.
AIIR_CAPI_EXPORTED bool aiirSMTTypeIsAnySMTValueType(AiirType type);

/// Checks if the given type is a smt::ArrayType.
AIIR_CAPI_EXPORTED bool aiirSMTTypeIsAArray(AiirType type);

/// Creates an array type with the given domain and range types.
AIIR_CAPI_EXPORTED AiirType aiirSMTTypeGetArray(AiirContext ctx,
                                                AiirType domainType,
                                                AiirType rangeType);

/// Checks if the given type is a smt::BitVectorType.
AIIR_CAPI_EXPORTED bool aiirSMTTypeIsABitVector(AiirType type);

/// Creates a smt::BitVectorType with the given width.
AIIR_CAPI_EXPORTED AiirType aiirSMTTypeGetBitVector(AiirContext ctx,
                                                    int32_t width);

AIIR_CAPI_EXPORTED AiirStringRef aiirSMTBitVectorTypeGetName(void);

AIIR_CAPI_EXPORTED AiirTypeID aiirSMTBitVectorTypeGetTypeID(void);

/// Checks if the given type is a smt::BoolType.
AIIR_CAPI_EXPORTED bool aiirSMTTypeIsABool(AiirType type);

/// Creates a smt::BoolType.
AIIR_CAPI_EXPORTED AiirType aiirSMTTypeGetBool(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirSMTBoolTypeGetName(void);

AIIR_CAPI_EXPORTED AiirTypeID aiirSMTBoolTypeGetTypeID(void);

/// Checks if the given type is a smt::IntType.
AIIR_CAPI_EXPORTED bool aiirSMTTypeIsAInt(AiirType type);

/// Creates a smt::IntType.
AIIR_CAPI_EXPORTED AiirType aiirSMTTypeGetInt(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirSMTIntTypeGetName(void);

AIIR_CAPI_EXPORTED AiirTypeID aiirSMTIntTypeGetTypeID(void);

/// Checks if the given type is a smt::FuncType.
AIIR_CAPI_EXPORTED bool aiirSMTTypeIsASMTFunc(AiirType type);

/// Creates a smt::FuncType with the given domain and range types.
AIIR_CAPI_EXPORTED AiirType aiirSMTTypeGetSMTFunc(AiirContext ctx,
                                                  size_t numberOfDomainTypes,
                                                  const AiirType *domainTypes,
                                                  AiirType rangeType);

/// Checks if the given type is a smt::SortType.
AIIR_CAPI_EXPORTED bool aiirSMTTypeIsASort(AiirType type);

/// Creates a smt::SortType with the given identifier and sort parameters.
AIIR_CAPI_EXPORTED AiirType aiirSMTTypeGetSort(AiirContext ctx,
                                               AiirIdentifier identifier,
                                               size_t numberOfSortParams,
                                               const AiirType *sortParams);

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

/// Checks if the given string is a valid smt::BVCmpPredicate.
AIIR_CAPI_EXPORTED bool aiirSMTAttrCheckBVCmpPredicate(AiirContext ctx,
                                                       AiirStringRef str);

/// Checks if the given string is a valid smt::IntPredicate.
AIIR_CAPI_EXPORTED bool aiirSMTAttrCheckIntPredicate(AiirContext ctx,
                                                     AiirStringRef str);

/// Checks if the given attribute is a smt::SMTAttribute.
AIIR_CAPI_EXPORTED bool aiirSMTAttrIsASMTAttribute(AiirAttribute attr);

/// Creates a smt::BitVectorAttr with the given value and width.
AIIR_CAPI_EXPORTED AiirAttribute aiirSMTAttrGetBitVector(AiirContext ctx,
                                                         uint64_t value,
                                                         unsigned width);

/// Creates a smt::BVCmpPredicateAttr with the given string.
AIIR_CAPI_EXPORTED AiirAttribute
aiirSMTAttrGetBVCmpPredicate(AiirContext ctx, AiirStringRef str);

/// Creates a smt::IntPredicateAttr with the given string.
AIIR_CAPI_EXPORTED AiirAttribute aiirSMTAttrGetIntPredicate(AiirContext ctx,
                                                            AiirStringRef str);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_DIALECT_SMT_H
