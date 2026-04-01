//===- PythonTestCAPI.cpp - C API for the PythonTest dialect --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PythonTestCAPI.h"
#include "PythonTestDialect.h"
#include "aiir-c/BuiltinTypes.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/CAPI/Wrap.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/Location.h"

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(PythonTest, python_test,
                                      python_test::PythonTestDialect)

bool aiirAttributeIsAPythonTestTestAttribute(AiirAttribute attr) {
  return llvm::isa<python_test::TestAttrAttr>(unwrap(attr));
}

AiirAttribute aiirPythonTestTestAttributeGet(AiirContext context) {
  return wrap(python_test::TestAttrAttr::get(unwrap(context)));
}

AiirTypeID aiirPythonTestTestAttributeGetTypeID(void) {
  return wrap(python_test::TestAttrAttr::getTypeID());
}

bool aiirTypeIsAPythonTestTestType(AiirType type) {
  return llvm::isa<python_test::TestTypeType>(unwrap(type));
}

AiirType aiirPythonTestTestTypeGet(AiirContext context) {
  return wrap(python_test::TestTypeType::get(unwrap(context)));
}

AiirTypeID aiirPythonTestTestTypeGetTypeID(void) {
  return wrap(python_test::TestTypeType::getTypeID());
}

bool aiirTypeIsAPythonTestTestTensorValue(AiirValue value) {
  return aiirTypeIsATensor(wrap(unwrap(value).getType()));
}

void aiirPythonTestEmitDiagnosticWithNote(AiirContext ctx) {
  auto diag =
      aiir::emitError(unwrap(aiirLocationUnknownGet(ctx)), "created error");
  diag.attachNote() << "attached note";
}
