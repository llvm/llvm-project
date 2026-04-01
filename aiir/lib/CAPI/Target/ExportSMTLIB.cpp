//===- ExportSMTLIB.cpp - C Interface to ExportSMTLIB ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements a C Interface for export SMTLIB.
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Target/ExportSMTLIB.h"
#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/Support.h"
#include "aiir/CAPI/Utils.h"
#include "aiir/Target/SMTLIB/ExportSMTLIB.h"

using namespace aiir;

AiirLogicalResult aiirTranslateOperationToSMTLIB(
    AiirOperation module, AiirStringCallback callback, void *userData,
    bool inlineSingleUseValues, bool indentLetBody, bool emitReset) {
  aiir::detail::CallbackOstream stream(callback, userData);
  smt::SMTEmissionOptions options;
  options.inlineSingleUseValues = inlineSingleUseValues;
  options.indentLetBody = indentLetBody;
  options.emitReset = emitReset;
  return wrap(smt::exportSMTLIB(unwrap(module), stream, options));
}

AiirLogicalResult
aiirTranslateModuleToSMTLIB(AiirModule module, AiirStringCallback callback,
                            void *userData, bool inlineSingleUseValues,
                            bool indentLetBody, bool emitReset) {
  return aiirTranslateOperationToSMTLIB(
      aiirModuleGetOperation(module), callback, userData, inlineSingleUseValues,
      indentLetBody, emitReset);
}
