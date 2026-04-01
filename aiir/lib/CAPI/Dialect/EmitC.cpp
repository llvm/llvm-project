//===- EmitC.cpp - C Interface for EmitC dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/EmitC.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/EmitC/IR/EmitC.h"

using namespace aiir;
using namespace aiir::emitc;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(EmitC, emitc, aiir::emitc::EmitCDialect)

// Ensure the C-API enums are uint64_t-castable to C++ equivalents.
static_assert(static_cast<uint64_t>(AIIR_EMITC_CMP_PREDICATE_EQ) ==
                      static_cast<uint64_t>(emitc::CmpPredicate::eq) &&
                  static_cast<uint64_t>(AIIR_EMITC_CMP_PREDICATE_NE) ==
                      static_cast<uint64_t>(emitc::CmpPredicate::ne) &&
                  static_cast<uint64_t>(AIIR_EMITC_CMP_PREDICATE_LT) ==
                      static_cast<uint64_t>(emitc::CmpPredicate::lt) &&
                  static_cast<uint64_t>(AIIR_EMITC_CMP_PREDICATE_LE) ==
                      static_cast<uint64_t>(emitc::CmpPredicate::le) &&
                  static_cast<uint64_t>(AIIR_EMITC_CMP_PREDICATE_GT) ==
                      static_cast<uint64_t>(emitc::CmpPredicate::gt) &&
                  static_cast<uint64_t>(AIIR_EMITC_CMP_PREDICATE_GE) ==
                      static_cast<uint64_t>(emitc::CmpPredicate::ge) &&
                  static_cast<uint64_t>(AIIR_EMITC_CMP_PREDICATE_THREE_WAY) ==
                      static_cast<uint64_t>(emitc::CmpPredicate::three_way),
              "AiirEmitCCmpPredicate (C-API) and CmpPredicate (C++) mismatch");

//===---------------------------------------------------------------------===//
// ArrayType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAEmitCArrayType(AiirType type) {
  return isa<emitc::ArrayType>(unwrap(type));
}

AiirTypeID aiirEmitCArrayTypeGetTypeID(void) {
  return wrap(emitc::ArrayType::getTypeID());
}

AiirType aiirEmitCArrayTypeGet(intptr_t nDims, int64_t *shape,
                               AiirType elementType) {
  return wrap(
      emitc::ArrayType::get(llvm::ArrayRef(shape, nDims), unwrap(elementType)));
}

AiirStringRef aiirEmitCArrayTypeGetName(void) {
  return wrap(emitc::ArrayType::name);
}

//===---------------------------------------------------------------------===//
// LValueType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAEmitCLValueType(AiirType type) {
  return isa<emitc::LValueType>(unwrap(type));
}

AiirTypeID aiirEmitCLValueTypeGetTypeID(void) {
  return wrap(emitc::LValueType::getTypeID());
}

AiirType aiirEmitCLValueTypeGet(AiirType valueType) {
  return wrap(emitc::LValueType::get(unwrap(valueType)));
}

AiirStringRef aiirEmitCLValueTypeGetName(void) {
  return wrap(emitc::LValueType::name);
}

//===---------------------------------------------------------------------===//
// OpaqueType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAEmitCOpaqueType(AiirType type) {
  return isa<emitc::OpaqueType>(unwrap(type));
}

AiirTypeID aiirEmitCOpaqueTypeGetTypeID(void) {
  return wrap(emitc::OpaqueType::getTypeID());
}

AiirType aiirEmitCOpaqueTypeGet(AiirContext ctx, AiirStringRef value) {
  return wrap(emitc::OpaqueType::get(unwrap(ctx), unwrap(value)));
}

AiirStringRef aiirEmitCOpaqueTypeGetName(void) {
  return wrap(emitc::OpaqueType::name);
}

//===---------------------------------------------------------------------===//
// PointerType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAEmitCPointerType(AiirType type) {
  return isa<emitc::PointerType>(unwrap(type));
}

AiirTypeID aiirEmitCPointerTypeGetTypeID(void) {
  return wrap(emitc::PointerType::getTypeID());
}

AiirType aiirEmitCPointerTypeGet(AiirType pointee) {
  return wrap(emitc::PointerType::get(unwrap(pointee)));
}

AiirStringRef aiirEmitCPointerTypeGetName(void) {
  return wrap(emitc::PointerType::name);
}

//===---------------------------------------------------------------------===//
// PtrDiffTType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAEmitCPtrDiffTType(AiirType type) {
  return isa<emitc::PtrDiffTType>(unwrap(type));
}

AiirTypeID aiirEmitCPtrDiffTTypeGetTypeID(void) {
  return wrap(emitc::PtrDiffTType::getTypeID());
}

AiirType aiirEmitCPtrDiffTTypeGet(AiirContext ctx) {
  return wrap(emitc::PtrDiffTType::get(unwrap(ctx)));
}

AiirStringRef aiirEmitCPtrDiffTTypeGetName(void) {
  return wrap(emitc::PtrDiffTType::name);
}

//===---------------------------------------------------------------------===//
// SignedSizeTType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAEmitCSignedSizeTType(AiirType type) {
  return isa<emitc::SignedSizeTType>(unwrap(type));
}

AiirTypeID aiirEmitCSignedSizeTTypeGetTypeID(void) {
  return wrap(emitc::SignedSizeTType::getTypeID());
}

AiirType aiirEmitCSignedSizeTTypeGet(AiirContext ctx) {
  return wrap(emitc::SignedSizeTType::get(unwrap(ctx)));
}

AiirStringRef aiirEmitCSignedSizeTTypeGetName(void) {
  return wrap(emitc::SignedSizeTType::name);
}

//===---------------------------------------------------------------------===//
// SizeTType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAEmitCSizeTType(AiirType type) {
  return isa<emitc::SizeTType>(unwrap(type));
}

AiirTypeID aiirEmitCSizeTTypeGetTypeID(void) {
  return wrap(emitc::SizeTType::getTypeID());
}

AiirType aiirEmitCSizeTTypeGet(AiirContext ctx) {
  return wrap(emitc::SizeTType::get(unwrap(ctx)));
}

AiirStringRef aiirEmitCSizeTTypeGetName(void) {
  return wrap(emitc::SizeTType::name);
}

//===----------------------------------------------------------------------===//
// CmpPredicate attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsAEmitCCmpPredicate(AiirAttribute attr) {
  return llvm::isa<emitc::CmpPredicateAttr>(unwrap(attr));
}

AiirAttribute aiirEmitCCmpPredicateAttrGet(AiirContext ctx,
                                           AiirEmitCCmpPredicate val) {
  return wrap((Attribute)emitc::CmpPredicateAttr::get(
      unwrap(ctx), static_cast<emitc::CmpPredicate>(val)));
}

AiirStringRef aiirEmitCCmpPredicateAttrGetName(void) {
  return wrap(emitc::CmpPredicateAttr::name);
}

AiirEmitCCmpPredicate aiirEmitCCmpPredicateAttrGetValue(AiirAttribute attr) {
  return static_cast<AiirEmitCCmpPredicate>(
      llvm::cast<emitc::CmpPredicateAttr>(unwrap(attr)).getValue());
}

AiirTypeID aiirEmitCCmpPredicateAttrGetTypeID(void) {
  return wrap(emitc::CmpPredicateAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Opaque attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsAEmitCOpaque(AiirAttribute attr) {
  return llvm::isa<emitc::OpaqueAttr>(unwrap(attr));
}

AiirAttribute aiirEmitCOpaqueAttrGet(AiirContext ctx, AiirStringRef value) {
  return wrap((Attribute)emitc::OpaqueAttr::get(unwrap(ctx), unwrap(value)));
}

AiirStringRef aiirEmitCOpaqueAttrGetName(void) {
  return wrap(emitc::OpaqueAttr::name);
}

AiirStringRef aiirEmitCOpaqueAttrGetValue(AiirAttribute attr) {
  return wrap(llvm::cast<emitc::OpaqueAttr>(unwrap(attr)).getValue());
}

AiirTypeID aiirEmitCOpaqueAttrGetTypeID(void) {
  return wrap(emitc::OpaqueAttr::getTypeID());
}
