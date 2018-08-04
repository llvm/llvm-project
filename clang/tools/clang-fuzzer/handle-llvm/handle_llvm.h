//==-- handle_llvm.h - Helper function for Clang fuzzers -------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines HandleLLVM for use by the Clang fuzzers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_CLANG_FUZZER_HANDLE_LLVM_HANDLELLVM_H
#define LLVM_CLANG_TOOLS_CLANG_FUZZER_HANDLE_LLVM_HANDLELLVM_H

#include <string>
#include <vector>

namespace clang_fuzzer {
void HandleLLVM(const std::string &S,
                const std::vector<const char *> &ExtraArgs);
} // namespace clang_fuzzer

#endif
