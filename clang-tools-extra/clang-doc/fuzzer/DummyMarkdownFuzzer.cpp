//===-- DummyMarkdownFuzzer.cpp - Entry point to test the fuzzer ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of main so we can build and test the harness without linking
// libFuzzer. Each command line argument is treated as a file to run the
// harness on.
//
//===----------------------------------------------------------------------===//

#include "llvm/FuzzMutate/FuzzerCLI.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);

int main(int argc, char *argv[]) {
  return llvm::runFuzzerOnInputs(argc, argv, LLVMFuzzerTestOneInput);
}
