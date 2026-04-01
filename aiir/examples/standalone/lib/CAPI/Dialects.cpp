//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone-c/Dialects.h"

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneTypes.h"
#include "aiir/CAPI/Registration.h"

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(Standalone, standalone,
                                      aiir::standalone::StandaloneDialect)

AiirType aiirStandaloneCustomTypeGet(AiirContext ctx, AiirStringRef value) {
  return wrap(aiir::standalone::CustomType::get(unwrap(ctx), unwrap(value)));
}

bool aiirStandaloneTypeIsACustomType(AiirType t) {
  return llvm::isa<aiir::standalone::CustomType>(unwrap(t));
}

AiirTypeID aiirStandaloneCustomTypeGetTypeID() {
  return wrap(aiir::standalone::CustomType::getTypeID());
}
