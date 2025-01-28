//===--- UseCppStyleCommentsCheck.h - clang-tidy-----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USECPPSTYLECOMMENTSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USECPPSTYLECOMMENTSCHECK_H

#include "../ClangTidyCheck.h"
#include <memory>

namespace clang::tidy::readability {
/// Detects C Style comments and suggests to use C++ style comments instead.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/use-cpp-style-comments.html
class UseCppStyleCommentsCheck : public ClangTidyCheck {
public:
  UseCppStyleCommentsCheck(StringRef Name, ClangTidyContext *Context);

  ~UseCppStyleCommentsCheck() override;

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;

  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  class CStyleCommentHandler;
  std::unique_ptr<CStyleCommentHandler> Handler;
};
} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USECPPSTYLECOMMENTSCHECK_H
