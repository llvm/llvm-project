//===-- examples/clang-interpreter/Test.cxx - Clang C Interpreter Example -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
