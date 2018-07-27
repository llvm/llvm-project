//===-- fuzzer_initialize.cpp - Fuzz Clang --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements two functions: one that returns the command line
/// arguments for a given call to the fuzz target and one that initializes
/// the fuzzer with the correct command line arguments.
///
//===----------------------------------------------------------------------===//

#include "fuzzer_initialize.h"

#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include <cstring>

using namespace clang_fuzzer;
using namespace llvm;


namespace clang_fuzzer {

static std::vector<const char *> CLArgs;

const std::vector<const char *>& GetCLArgs() {
  return CLArgs;
}

}

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();
  
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeScalarOpts(Registry);
  initializeVectorization(Registry);
  initializeIPO(Registry);
  initializeAnalysis(Registry);
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  initializeAggressiveInstCombine(Registry);
  initializeInstrumentation(Registry);
  initializeTarget(Registry);

  CLArgs.push_back("-O2");
  for (int I = 1; I < *argc; I++) {
    if (strcmp((*argv)[I], "-ignore_remaining_args=1") == 0) {
      for (I++; I < *argc; I++)
        CLArgs.push_back((*argv)[I]);
      break;
    }
  }
  return 0;
}
