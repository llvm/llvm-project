//===- InitAllTranslations.h - AIIR Translations Registration ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all translations
// in and out of AIIR to the system.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_INITALLTRANSLATIONS_H
#define AIIR_INITALLTRANSLATIONS_H

#include "aiir/Target/IRDLToCpp/TranslationRegistration.h"

namespace aiir {

void registerFromLLVMIRTranslation();
void registerFromSPIRVTranslation();
void registerFromWasmTranslation();
void registerToCppTranslation();
void registerToLLVMIRTranslation();
void registerToSPIRVTranslation();

namespace smt {
void registerExportSMTLIBTranslation();
}

// This function should be called before creating any AIIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerAllTranslations() {
  static bool initOnce = []() {
    registerFromLLVMIRTranslation();
    registerFromSPIRVTranslation();
    registerIRDLToCppTranslation();
    registerFromWasmTranslation();
    registerToCppTranslation();
    registerToLLVMIRTranslation();
    registerToSPIRVTranslation();
    smt::registerExportSMTLIBTranslation();
    return true;
  }();
  (void)initOnce;
}
} // namespace aiir

#endif // AIIR_INITALLTRANSLATIONS_H
