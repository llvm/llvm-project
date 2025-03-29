//===--- ImplicitWideningOfMultiplicationResultCheck.h ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_IMPLICITWIDENINGOFMULTIPLICATIONRESULTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_IMPLICITWIDENINGOFMULTIPLICATIONRESULTCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"
#include <optional>

namespace clang::tidy::bugprone {

/// Diagnoses instances of an implicit widening of multiplication result.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/implicit-widening-of-multiplication-result.html
class ImplicitWideningOfMultiplicationResultCheck : public ClangTidyCheck {
  const ast_matchers::MatchFinder::MatchResult *Result;
  bool ShouldUseCXXStaticCast;
  bool ShouldUseCXXHeader;

  std::optional<FixItHint> includeStddefHeader(SourceLocation File);

  void handleImplicitCastExpr(const ImplicitCastExpr *ICE);
  void handlePointerOffsetting(const Expr *E);

public:
  ImplicitWideningOfMultiplicationResultCheck(StringRef Name,
                                              ClangTidyContext *Context);
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const bool UseCXXStaticCastsInCppSources;
  const bool UseCXXHeadersInCppSources;
  const bool IgnoreConstantIntExpr;
  utils::IncludeInserter IncludeInserter;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_IMPLICITWIDENINGOFMULTIPLICATIONRESULTCHECK_H
