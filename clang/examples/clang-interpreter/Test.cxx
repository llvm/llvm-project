//===-- examples/clang-interpreter/Test.cxx - Clang C Interpreter Example -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Example throwing in and from the JIT (particularly on Win64).
//
// ./bin/clang-interpreter <src>/tools/clang/examples/clang-interpreter/Test.cxx

#include <stdexcept>
#include <stdio.h>

static void ThrowerAnError(const char* Name) {
  throw std::runtime_error(Name);
}

int main(int argc, const char** argv) {
  for (int I = 0; I < argc; ++I)
   printf("arg[%d]='%s'\n", I, argv[I]);

  try {
    ThrowerAnError("In JIT");
  } catch (const std::exception& E) {
    printf("Caught: '%s'\n", E.what());
  } catch (...) {
    printf("Unknown exception\n");
  }
  ThrowerAnError("From JIT");
  return 0;
}
