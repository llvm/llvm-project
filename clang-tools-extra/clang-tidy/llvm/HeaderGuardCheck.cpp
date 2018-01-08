//===--- HeaderGuardCheck.cpp - clang-tidy --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HeaderGuardCheck.h"

namespace clang {
namespace tidy {
namespace llvm {

LLVMHeaderGuardCheck::LLVMHeaderGuardCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : HeaderGuardCheck(Name, Context) {}

std::string LLVMHeaderGuardCheck::getHeaderGuard(StringRef Filename,
                                                 StringRef OldGuard) {
  std::string Guard = tooling::getAbsolutePath(Filename);

  // Sanitize the path. There are some rules for compatibility with the historic
  // style in include/llvm and include/clang which we want to preserve.

  // We don't want _INCLUDE_ in our guards.
  size_t PosInclude = Guard.rfind("include/");
  if (PosInclude != StringRef::npos)
    Guard = Guard.substr(PosInclude + std::strlen("include/"));

  // For clang we drop the _TOOLS_.
  size_t PosToolsClang = Guard.rfind("tools/clang/");
  if (PosToolsClang != StringRef::npos)
    Guard = Guard.substr(PosToolsClang + std::strlen("tools/"));

  // The remainder is LLVM_FULL_PATH_TO_HEADER_H
  size_t PosLLVM = Guard.rfind("llvm/");
  if (PosLLVM != StringRef::npos)
    Guard = Guard.substr(PosLLVM);

  std::replace(Guard.begin(), Guard.end(), '/', '_');
  std::replace(Guard.begin(), Guard.end(), '.', '_');
  std::replace(Guard.begin(), Guard.end(), '-', '_');

  // The prevalent style in clang is LLVM_CLANG_FOO_BAR_H
  if (StringRef(Guard).startswith("clang"))
    Guard = "LLVM_" + Guard;

  return StringRef(Guard).upper();
}

} // namespace llvm
} // namespace tidy
} // namespace clang
