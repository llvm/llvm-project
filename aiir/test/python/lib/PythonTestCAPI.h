//===- PythonTestCAPI.h - C API for the PythonTest dialect ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TEST_PYTHON_LIB_PYTHONTESTCAPI_H
#define AIIR_TEST_PYTHON_LIB_PYTHONTESTCAPI_H

#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(PythonTest, python_test);

AIIR_CAPI_EXPORTED bool
aiirAttributeIsAPythonTestTestAttribute(AiirAttribute attr);

AIIR_CAPI_EXPORTED AiirAttribute
aiirPythonTestTestAttributeGet(AiirContext context);

AIIR_CAPI_EXPORTED AiirTypeID aiirPythonTestTestAttributeGetTypeID(void);

AIIR_CAPI_EXPORTED bool aiirTypeIsAPythonTestTestType(AiirType type);

AIIR_CAPI_EXPORTED AiirType aiirPythonTestTestTypeGet(AiirContext context);

AIIR_CAPI_EXPORTED AiirTypeID aiirPythonTestTestTypeGetTypeID(void);

AIIR_CAPI_EXPORTED bool aiirTypeIsAPythonTestTestTensorValue(AiirValue value);

AIIR_CAPI_EXPORTED void aiirPythonTestEmitDiagnosticWithNote(AiirContext ctx);

#ifdef __cplusplus
}
#endif

#endif // AIIR_TEST_PYTHON_LIB_PYTHONTESTCAPI_H
