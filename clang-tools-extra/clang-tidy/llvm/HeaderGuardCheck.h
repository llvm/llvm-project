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
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/llvm-header-guard.html
/// The check supports these options:
///   - `HeaderFileExtensions`: a comma-separated list of filename extensions of
///     header files (The filename extension should not contain "." prefix).
///     ",h,hh,hpp,hxx" by default.
///     For extension-less header files, using an empty string or leaving an
///     empty string between "," if there are other filename extensions.
class LLVMHeaderGuardCheck : public utils::HeaderGuardCheck {
public:
  LLVMHeaderGuardCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool shouldSuggestEndifComment(StringRef Filename) override { return false; }
  bool shouldFixHeaderGuard(StringRef Filename) override;
  std::string getHeaderGuard(StringRef Filename, StringRef OldGuard) override;

private:
  std::string RawStringHeaderFileExtensions;
  utils::HeaderFileExtensionsSet HeaderFileExtensions;
};

} // namespace llvm
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_HEADER_GUARD_CHECK_H
