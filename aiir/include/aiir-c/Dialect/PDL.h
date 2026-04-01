//===-- aiir-c/Dialect/PDL.h - C API for PDL Dialect --------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_PDL_H
#define AIIR_C_DIALECT_PDL_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(PDL, pdl);

//===---------------------------------------------------------------------===//
// PDLType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAPDLType(AiirType type);

//===---------------------------------------------------------------------===//
// AttributeType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAPDLAttributeType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirPDLAttributeTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirPDLAttributeTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirPDLAttributeTypeGetName(void);

//===---------------------------------------------------------------------===//
// OperationType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAPDLOperationType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirPDLOperationTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirPDLOperationTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirPDLOperationTypeGetName(void);

//===---------------------------------------------------------------------===//
// RangeType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAPDLRangeType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirPDLRangeTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirPDLRangeTypeGet(AiirType elementType);

AIIR_CAPI_EXPORTED AiirStringRef aiirPDLRangeTypeGetName(void);

AIIR_CAPI_EXPORTED AiirType aiirPDLRangeTypeGetElementType(AiirType type);

//===---------------------------------------------------------------------===//
// TypeType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAPDLTypeType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirPDLTypeTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirPDLTypeTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirPDLTypeTypeGetName(void);

//===---------------------------------------------------------------------===//
// ValueType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAPDLValueType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirPDLValueTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirType aiirPDLValueTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirPDLValueTypeGetName(void);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_DIALECT_PDL_H
