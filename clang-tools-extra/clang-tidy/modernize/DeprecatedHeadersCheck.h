//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_C_HEADERS_TO_CXX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_C_HEADERS_TO_CXX_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// This check replaces deprecated C library headers with their C++ STL
/// alternatives.
///
/// Before:
/// ~~~{.cpp}
/// #include <header.h>
/// ~~~
///
/// After:
/// ~~~{.cpp}
/// #include <cheader>
/// ~~~
///
/// Example: ``<stdio.h> => <cstdio>``
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/modernize/deprecated-headers.html
class DeprecatedHeadersCheck : public ClangTidyCheck {
public:
  DeprecatedHeadersCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void onEndOfTranslationUnit() override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  struct IncludeMarker {
    StringRef Replacement;
    StringRef FileName;
    SourceRange ReplacementRange;
    SourceLocation DiagLoc;
  };

private:
  std::vector<IncludeMarker> IncludesToBeProcessed;
  bool CheckHeaderFile;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_C_HEADERS_TO_CXX_H
