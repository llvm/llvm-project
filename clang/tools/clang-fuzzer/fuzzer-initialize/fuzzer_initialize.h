//==-- fuzzer_initialize.h - Fuzz Clang ------------------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines a function that returns the command line arguments for a specific
// call to the fuzz target.
//
//===----------------------------------------------------------------------===//

#include <vector>

namespace clang_fuzzer {
const std::vector<const char *>& GetCLArgs();
}
