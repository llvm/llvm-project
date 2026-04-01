//===- SMT.cpp - C interface for the SMT dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/SMT.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/SMT/IR/SMTAttributes.h"
#include "aiir/Dialect/SMT/IR/SMTDialect.h"
#include "aiir/Dialect/SMT/IR/SMTTypes.h"

using namespace aiir;
using namespace smt;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(SMT, smt, aiir::smt::SMTDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

bool aiirSMTTypeIsAnyNonFuncSMTValueType(AiirType type) {
  return isAnyNonFuncSMTValueType(unwrap(type));
}

bool aiirSMTTypeIsAnySMTValueType(AiirType type) {
  return isAnySMTValueType(unwrap(type));
}

bool aiirSMTTypeIsAArray(AiirType type) { return isa<ArrayType>(unwrap(type)); }

AiirType aiirSMTTypeGetArray(AiirContext ctx, AiirType domainType,
                             AiirType rangeType) {
  return wrap(
      ArrayType::get(unwrap(ctx), unwrap(domainType), unwrap(rangeType)));
}

bool aiirSMTTypeIsABitVector(AiirType type) {
  return isa<BitVectorType>(unwrap(type));
}

AiirType aiirSMTTypeGetBitVector(AiirContext ctx, int32_t width) {
  return wrap(BitVectorType::get(unwrap(ctx), width));
}

AiirStringRef aiirSMTBitVectorTypeGetName(void) {
  return wrap(BitVectorType::name);
}

AiirTypeID aiirSMTBitVectorTypeGetTypeID(void) {
  return wrap(BitVectorType::getTypeID());
}

bool aiirSMTTypeIsABool(AiirType type) { return isa<BoolType>(unwrap(type)); }

AiirType aiirSMTTypeGetBool(AiirContext ctx) {
  return wrap(BoolType::get(unwrap(ctx)));
}

AiirStringRef aiirSMTBoolTypeGetName(void) { return wrap(BoolType::name); }

AiirTypeID aiirSMTBoolTypeGetTypeID(void) {
  return wrap(BoolType::getTypeID());
}

bool aiirSMTTypeIsAInt(AiirType type) { return isa<IntType>(unwrap(type)); }

AiirType aiirSMTTypeGetInt(AiirContext ctx) {
  return wrap(IntType::get(unwrap(ctx)));
}

AiirStringRef aiirSMTIntTypeGetName(void) { return wrap(IntType::name); }

AiirTypeID aiirSMTIntTypeGetTypeID(void) { return wrap(IntType::getTypeID()); }

bool aiirSMTTypeIsASMTFunc(AiirType type) {
  return isa<SMTFuncType>(unwrap(type));
}

AiirType aiirSMTTypeGetSMTFunc(AiirContext ctx, size_t numberOfDomainTypes,
                               const AiirType *domainTypes,
                               AiirType rangeType) {
  SmallVector<Type> domainTypesVec;
  domainTypesVec.reserve(numberOfDomainTypes);

  for (size_t i = 0; i < numberOfDomainTypes; i++)
    domainTypesVec.push_back(unwrap(domainTypes[i]));

  return wrap(SMTFuncType::get(unwrap(ctx), domainTypesVec, unwrap(rangeType)));
}

bool aiirSMTTypeIsASort(AiirType type) { return isa<SortType>(unwrap(type)); }

AiirType aiirSMTTypeGetSort(AiirContext ctx, AiirIdentifier identifier,
                            size_t numberOfSortParams,
                            const AiirType *sortParams) {
  SmallVector<Type> sortParamsVec;
  sortParamsVec.reserve(numberOfSortParams);

  for (size_t i = 0; i < numberOfSortParams; i++)
    sortParamsVec.push_back(unwrap(sortParams[i]));

  return wrap(SortType::get(unwrap(ctx), unwrap(identifier), sortParamsVec));
}

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

bool aiirSMTAttrCheckBVCmpPredicate(AiirContext ctx, AiirStringRef str) {
  return symbolizeBVCmpPredicate(unwrap(str)).has_value();
}

bool aiirSMTAttrCheckIntPredicate(AiirContext ctx, AiirStringRef str) {
  return symbolizeIntPredicate(unwrap(str)).has_value();
}

bool aiirSMTAttrIsASMTAttribute(AiirAttribute attr) {
  return isa<BitVectorAttr, BVCmpPredicateAttr, IntPredicateAttr>(unwrap(attr));
}

AiirAttribute aiirSMTAttrGetBitVector(AiirContext ctx, uint64_t value,
                                      unsigned width) {
  return wrap(BitVectorAttr::get(unwrap(ctx), value, width));
}

AiirAttribute aiirSMTAttrGetBVCmpPredicate(AiirContext ctx, AiirStringRef str) {
  auto predicate = symbolizeBVCmpPredicate(unwrap(str));
  assert(predicate.has_value() && "invalid predicate");

  return wrap(BVCmpPredicateAttr::get(unwrap(ctx), predicate.value()));
}

AiirAttribute aiirSMTAttrGetIntPredicate(AiirContext ctx, AiirStringRef str) {
  auto predicate = symbolizeIntPredicate(unwrap(str));
  assert(predicate.has_value() && "invalid predicate");

  return wrap(IntPredicateAttr::get(unwrap(ctx), predicate.value()));
}
