//===-- DummyClangFuzzer.cpp - Entry point to sanity check fuzzers --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Provides a main() to build without linking libFuzzer.
//
//===----------------------------------------------------------------------===//
#include "llvm/FuzzMutate/FuzzerCLI.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);
extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv);

int main(int argc, char *argv[]) {
  return llvm::runFuzzerOnInputs(argc, argv, LLVMFuzzerTestOneInput,
                                 LLVMFuzzerInitialize);
}
