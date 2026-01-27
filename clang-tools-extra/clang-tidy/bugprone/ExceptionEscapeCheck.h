//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_EXCEPTIONESCAPECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_EXCEPTIONESCAPECHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/ExceptionAnalyzer.h"
#include "llvm/ADT/StringSet.h"

namespace clang::tidy::bugprone {

/// Finds functions which should not throw exceptions: Destructors, move
/// constructors, move assignment operators, the main() function,
/// swap() functions, functions marked with throw() or noexcept and functions
/// given as option to the checker.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/exception-escape.html
class ExceptionEscapeCheck : public ClangTidyCheck {
public:
  ExceptionEscapeCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus && LangOpts.CXXExceptions;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  StringRef RawFunctionsThatShouldNotThrow;
  StringRef RawIgnoredExceptions;
  StringRef RawCheckedSwapFunctions;

  const bool CheckDestructors;
  const bool CheckMoveMemberFunctions;
  const bool CheckMain;
  const bool CheckNothrowFunctions;

  llvm::StringSet<> FunctionsThatShouldNotThrow;
  llvm::StringSet<> CheckedSwapFunctions;
  utils::ExceptionAnalyzer Tracer;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_EXCEPTIONESCAPECHECK_H
