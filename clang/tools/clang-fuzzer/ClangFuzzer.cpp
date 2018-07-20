//===-- ClangFuzzer.cpp - Fuzz Clang --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a function that runs Clang on a single
///  input. This function is then linked into the Fuzzer library.
///
//===----------------------------------------------------------------------===//

#include "handle-cxx/handle_cxx.h"

using namespace clang_fuzzer;

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) { return 0; }

extern "C" int LLVMFuzzerTestOneInput(uint8_t *data, size_t size) {
  std::string s((const char *)data, size);
  HandleCXX(s, {"-O2"});
  return 0;
}
