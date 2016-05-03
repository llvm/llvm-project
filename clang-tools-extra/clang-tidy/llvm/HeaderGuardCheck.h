//===--- HeaderGuardCheck.h - clang-tidy ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_HEADER_GUARD_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_HEADER_GUARD_CHECK_H

#include "../utils/HeaderGuard.h"

namespace clang {
namespace tidy {
namespace llvm {

/// Finds and fixes header guards that do not adhere to LLVM style.
class LLVMHeaderGuardCheck : public utils::HeaderGuardCheck {
public:
  LLVMHeaderGuardCheck(StringRef Name, ClangTidyContext *Context)
      : HeaderGuardCheck(Name, Context) {}
  bool shouldSuggestEndifComment(StringRef Filename) override { return false; }
  bool shouldFixHeaderGuard(StringRef Filename) override;
  std::string getHeaderGuard(StringRef Filename, StringRef OldGuard) override;
};

} // namespace llvm
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_HEADER_GUARD_CHECK_H
