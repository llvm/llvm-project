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

/// Transforms the repeated calls to `std::min` and `std::max` into a single
/// call using initializer lists.
///
/// It identifies cases where `std::min` or `std::max` is used to find the
/// minimum or maximum value among more than two items through repeated calls.
/// The check replaces these calls with a single call to `std::min` or
/// `std::max` that uses an initializer list. This makes the code slightly more
/// efficient.
/// \n\n
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

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  utils::IncludeInserter Inserter;
  void findArgs(const CallExpr *call, const Expr **first, const Expr **last,
                std::vector<const Expr *> &args);
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MINMAXUSEINITIALIZERLISTCHECK_H
