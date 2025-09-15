//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_RVALUEREFERENCEPARAMNOTMOVEDCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_RVALUEREFERENCEPARAMNOTMOVEDCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::cppcoreguidelines {

/// Warns when an rvalue reference function parameter is never moved within
/// the function body. This check implements CppCoreGuideline F.18.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines/rvalue-reference-param-not-moved.html
class RvalueReferenceParamNotMovedCheck : public ClangTidyCheck {
public:
  RvalueReferenceParamNotMovedCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const bool AllowPartialMove;
  const bool IgnoreUnnamedParams;
  const bool IgnoreNonDeducedTemplateTypes;
  const StringRef MoveFunction;
};

} // namespace clang::tidy::cppcoreguidelines

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_RVALUEREFERENCEPARAMNOTMOVEDCHECK_H
