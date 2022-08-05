//===--- SuspiciousIncludeCheck.h - clang-tidy ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSINCLUDECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSINCLUDECHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/FileExtensionsUtils.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Warns on inclusion of files whose names suggest that they're implementation
/// files, instead of headers. E.g:
///
///     #include "foo.cpp"  // warning
///     #include "bar.c"    // warning
///     #include "baz.h"    // no diagnostic
///
/// The check supports these options:
///   - `HeaderFileExtensions`: a semicolon-separated list of filename
///     extensions of header files (The filename extensions should not contain
///     "." prefix) ";h;hh;hpp;hxx" by default. For extension-less header
///     files, using an empty string or leaving an empty string between ";" if
///     there are other filename extensions.
///
///   - `ImplementationFileExtensions`: likewise, a semicolon-separated list of
///     filename extensions of implementation files. "c;cc;cpp;cxx" by default.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/suspicious-include.html
class SuspiciousIncludeCheck : public ClangTidyCheck {
public:
  SuspiciousIncludeCheck(StringRef Name, ClangTidyContext *Context);
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  utils::FileExtensionsSet HeaderFileExtensions;
  utils::FileExtensionsSet ImplementationFileExtensions;

private:
  const std::string RawStringHeaderFileExtensions;
  const std::string RawStringImplementationFileExtensions;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SUSPICIOUSINCLUDECHECK_H
