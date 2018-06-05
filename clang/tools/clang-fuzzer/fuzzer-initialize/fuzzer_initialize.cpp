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
#include <cstring>

using namespace clang_fuzzer;


namespace clang_fuzzer {

static std::vector<const char *> CLArgs;

const std::vector<const char *>& GetCLArgs() {
  return CLArgs;
}

}

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
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
