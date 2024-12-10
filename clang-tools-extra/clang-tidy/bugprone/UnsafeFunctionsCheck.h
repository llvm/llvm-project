//===--- UnsafeFunctionsCheck.h - clang-tidy --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNSAFEFUNCTIONSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNSAFEFUNCTIONSCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/Matchers.h"
#include <optional>

namespace clang::tidy::bugprone {

/// Checks for functions that have safer, more secure replacements available, or
/// are considered deprecated due to design flaws. This check relies heavily on,
/// but is not exclusive to, the functions from the
/// Annex K. "Bounds-checking interfaces" of C11.
///
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/unsafe-functions.html
class UnsafeFunctionsCheck : public ClangTidyCheck {
public:
  UnsafeFunctionsCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void onEndOfTranslationUnit() override;

  struct CheckedFunction {
    std::string Name;
    matchers::MatchesAnyListedNameMatcher::NameMatcher Pattern;
    std::string Replacement;
    std::string Reason;
  };

private:
  const std::vector<CheckedFunction> CustomFunctions;

  // If true, the default set of functions are reported.
  const bool ReportDefaultFunctions;
  /// If true, additional functions from widely used API-s (such as POSIX) are
  /// added to the list of reported functions.
  const bool ReportMoreUnsafeFunctions;

  Preprocessor *PP = nullptr;
  /// Whether "Annex K" functions are available and should be
  /// suggested in diagnostics. This is filled and cached internally.
  std::optional<bool> IsAnnexKAvailable;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNSAFEFUNCTIONSCHECK_H
