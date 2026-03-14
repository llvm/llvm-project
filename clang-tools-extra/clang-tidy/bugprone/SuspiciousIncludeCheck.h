//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSINCLUDECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSINCLUDECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Warns on inclusion of files whose names suggest that they're implementation
/// files, instead of headers. E.g:
///
///     #include "foo.cpp"  // warning
///     #include "bar.c"    // warning
///     #include "baz.h"    // no diagnostic
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/suspicious-include.html
class SuspiciousIncludeCheck : public ClangTidyCheck {
public:
  SuspiciousIncludeCheck(StringRef Name, ClangTidyContext *Context);
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  FileExtensionsSet HeaderFileExtensions;
  FileExtensionsSet ImplementationFileExtensions;
  StringRef IgnoredRegexString;
  llvm::Regex IgnoredRegex;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSINCLUDECHECK_H
