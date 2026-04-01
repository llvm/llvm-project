//===- aiir-translate.cpp - AIIR Translate Driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to AIIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "aiir/InitAllTranslations.h"
#include "aiir/Support/LLVM.h"
#include "aiir/Tools/aiir-translate/AiirTranslateMain.h"

using namespace aiir;

namespace aiir {
// Defined in the test directory, no public header.
void registerTestRoundtripSPIRV();
void registerTestRoundtripDebugSPIRV();
#ifdef AIIR_INCLUDE_TESTS
void registerTestToLLVMIR();
void registerTestFromLLVMIR();
#endif
} // namespace aiir

static void registerTestTranslations() {
  registerTestRoundtripSPIRV();
  registerTestRoundtripDebugSPIRV();
#ifdef AIIR_INCLUDE_TESTS
  registerTestToLLVMIR();
  registerTestFromLLVMIR();
#endif
}

int main(int argc, char **argv) {
  registerAllTranslations();
  registerTestTranslations();
  return failed(aiirTranslateMain(argc, argv, "AIIR Translation Testing Tool"));
}
