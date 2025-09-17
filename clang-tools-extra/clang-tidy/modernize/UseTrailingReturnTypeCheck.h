//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USETRAILINGRETURNTYPECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USETRAILINGRETURNTYPECHECK_H

#include "../ClangTidyCheck.h"
#include "clang/Lex/Token.h"

namespace clang::tidy::modernize {

struct ClassifiedToken {
  Token T;
  bool IsQualifier;
  bool IsSpecifier;
};

/// Rewrites function signatures to use a trailing return type.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/use-trailing-return-type.html
class UseTrailingReturnTypeCheck : public ClangTidyCheck {
public:
  UseTrailingReturnTypeCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  enum TransformLambda { All, AllExceptAuto, None };

private:
  Preprocessor *PP = nullptr;
  const bool TransformFunctions;
  const TransformLambda TransformLambdas;

  void diagOnLambda(const LambdaExpr *Lambda,
                    const ast_matchers::MatchFinder::MatchResult &Result);
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USETRAILINGRETURNTYPECHECK_H
