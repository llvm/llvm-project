//===--- ExceptionEscapeCheck.h - clang-tidy --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OPENMP_EXCEPTIONESCAPECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OPENMP_EXCEPTIONESCAPECHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/ExceptionAnalyzer.h"

namespace clang {
namespace tidy {
namespace openmp {

/// Analyzes OpenMP Structured Blocks and checks that no exception escapes
/// out of the Structured Block it was thrown in.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/openmp-exception-escape.html
class ExceptionEscapeCheck : public ClangTidyCheck {
public:
  ExceptionEscapeCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.OpenMP && LangOpts.CPlusPlus && LangOpts.CXXExceptions;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  std::string RawIgnoredExceptions;

  utils::ExceptionAnalyzer Tracer;
};

} // namespace openmp
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_OPENMP_EXCEPTIONESCAPECHECK_H
