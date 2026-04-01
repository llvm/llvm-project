//===- IRDL.cpp - C Interface for IRDL dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/IRDL.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/IRDL/IR/IRDL.h"
#include "aiir/Dialect/IRDL/IRDLLoading.h"

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(IRDL, irdl, aiir::irdl::IRDLDialect)

AiirLogicalResult aiirLoadIRDLDialects(AiirModule module) {
  return wrap(aiir::irdl::loadDialects(unwrap(module)));
}

//===----------------------------------------------------------------------===//
// VariadicityAttr
//===----------------------------------------------------------------------===//

AiirAttribute aiirIRDLVariadicityAttrGet(AiirContext ctx, AiirStringRef value) {
  return wrap(aiir::irdl::VariadicityAttr::get(
      unwrap(ctx), aiir::irdl::symbolizeVariadicity(unwrap(value)).value()));
}

AiirStringRef aiirIRDLVariadicityAttrGetName(void) {
  return wrap(aiir::irdl::VariadicityAttr::name);
}

//===----------------------------------------------------------------------===//
// VariadicityArrayAttr
//===----------------------------------------------------------------------===//

AiirAttribute aiirIRDLVariadicityArrayAttrGet(AiirContext ctx, intptr_t nValues,
                                              AiirAttribute const *values) {
  llvm::SmallVector<aiir::Attribute> attrs;
  llvm::ArrayRef<aiir::Attribute> unwrappedAttrs =
      unwrapList(nValues, values, attrs);

  llvm::SmallVector<aiir::irdl::VariadicityAttr> variadicities;
  for (auto attr : unwrappedAttrs)
    variadicities.push_back(llvm::cast<aiir::irdl::VariadicityAttr>(attr));

  return wrap(
      aiir::irdl::VariadicityArrayAttr::get(unwrap(ctx), variadicities));
}

AiirStringRef aiirIRDLVariadicityArrayAttrGetName(void) {
  return wrap(aiir::irdl::VariadicityArrayAttr::name);
}
