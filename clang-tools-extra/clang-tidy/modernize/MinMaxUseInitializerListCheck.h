//===--- MinMaxUseInitializerListCheck.h - clang-tidy -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MINMAXUSEINITIALIZERLISTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MINMAXUSEINITIALIZERLISTCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"

namespace clang::tidy::modernize {

/// Replaces chained ``std::min`` and ``std::max`` calls with a initializer list
/// where applicable.
///
/// For example:
///
/// \code
///   int a = std::max(std::max(i, j), k);
/// \endcode
///
/// This code is transformed to:
///
/// \code
///   int a = std::max({i, j, k});
/// \endcode
class MinMaxUseInitializerListCheck : public ClangTidyCheck {
public:
  MinMaxUseInitializerListCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Match) override;

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  struct FindArgsResult {
    const Expr *First;
    const Expr *Last;
    const Expr *Compare;
    std::vector<const Expr *> Args;
  };
  utils::IncludeInserter Inserter;
  FindArgsResult findArgs(const ast_matchers::MatchFinder::MatchResult &Match,
                          const CallExpr *Call);
  std::string
  generateReplacement(const ast_matchers::MatchFinder::MatchResult &Match,
                      const CallExpr *TopCall, const FindArgsResult Result);
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MINMAXUSEINITIALIZERLISTCHECK_H
