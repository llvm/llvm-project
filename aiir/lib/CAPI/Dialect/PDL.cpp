//===- PDL.cpp - C Interface for PDL dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/PDL.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/PDL/IR/PDL.h"
#include "aiir/Dialect/PDL/IR/PDLOps.h"
#include "aiir/Dialect/PDL/IR/PDLTypes.h"

using namespace aiir;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(PDL, pdl, pdl::PDLDialect)

//===---------------------------------------------------------------------===//
// PDLType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAPDLType(AiirType type) {
  return isa<pdl::PDLType>(unwrap(type));
}

//===---------------------------------------------------------------------===//
// AttributeType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAPDLAttributeType(AiirType type) {
  return isa<pdl::AttributeType>(unwrap(type));
}

AiirTypeID aiirPDLAttributeTypeGetTypeID(void) {
  return wrap(pdl::AttributeType::getTypeID());
}

AiirType aiirPDLAttributeTypeGet(AiirContext ctx) {
  return wrap(pdl::AttributeType::get(unwrap(ctx)));
}

AiirStringRef aiirPDLAttributeTypeGetName(void) {
  return wrap(pdl::AttributeType::name);
}

//===---------------------------------------------------------------------===//
// OperationType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAPDLOperationType(AiirType type) {
  return isa<pdl::OperationType>(unwrap(type));
}

AiirTypeID aiirPDLOperationTypeGetTypeID(void) {
  return wrap(pdl::OperationType::getTypeID());
}

AiirType aiirPDLOperationTypeGet(AiirContext ctx) {
  return wrap(pdl::OperationType::get(unwrap(ctx)));
}

AiirStringRef aiirPDLOperationTypeGetName(void) {
  return wrap(pdl::OperationType::name);
}

//===---------------------------------------------------------------------===//
// RangeType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAPDLRangeType(AiirType type) {
  return isa<pdl::RangeType>(unwrap(type));
}

AiirTypeID aiirPDLRangeTypeGetTypeID(void) {
  return wrap(pdl::RangeType::getTypeID());
}

AiirType aiirPDLRangeTypeGet(AiirType elementType) {
  return wrap(pdl::RangeType::get(unwrap(elementType)));
}

AiirStringRef aiirPDLRangeTypeGetName(void) {
  return wrap(pdl::RangeType::name);
}

AiirType aiirPDLRangeTypeGetElementType(AiirType type) {
  return wrap(cast<pdl::RangeType>(unwrap(type)).getElementType());
}

//===---------------------------------------------------------------------===//
// TypeType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAPDLTypeType(AiirType type) {
  return isa<pdl::TypeType>(unwrap(type));
}

AiirTypeID aiirPDLTypeTypeGetTypeID(void) {
  return wrap(pdl::TypeType::getTypeID());
}

AiirType aiirPDLTypeTypeGet(AiirContext ctx) {
  return wrap(pdl::TypeType::get(unwrap(ctx)));
}

AiirStringRef aiirPDLTypeTypeGetName(void) { return wrap(pdl::TypeType::name); }

//===---------------------------------------------------------------------===//
// ValueType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAPDLValueType(AiirType type) {
  return isa<pdl::ValueType>(unwrap(type));
}

AiirTypeID aiirPDLValueTypeGetTypeID(void) {
  return wrap(pdl::ValueType::getTypeID());
}

AiirType aiirPDLValueTypeGet(AiirContext ctx) {
  return wrap(pdl::ValueType::get(unwrap(ctx)));
}

AiirStringRef aiirPDLValueTypeGetName(void) {
  return wrap(pdl::ValueType::name);
}
