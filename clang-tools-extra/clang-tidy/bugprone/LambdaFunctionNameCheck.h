//===--- LambdaFunctionNameCheck.h - clang-tidy------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_LAMBDAFUNCTIONNAMECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_LAMBDAFUNCTIONNAMECHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Detect when __func__ or __FUNCTION__ is being used from within a lambda. In
/// that context, those expressions expand to the name of the call operator
/// (i.e., `operator()`).
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone-lambda-function-name.html
class LambdaFunctionNameCheck : public ClangTidyCheck {
public:
  struct SourceRangeLessThan {
    bool operator()(const SourceRange &L, const SourceRange &R) const {
      if (L.getBegin() == R.getBegin()) {
        return L.getEnd() < R.getEnd();
      }
      return L.getBegin() < R.getBegin();
    }
  };
  using SourceRangeSet = std::set<SourceRange, SourceRangeLessThan>;

  LambdaFunctionNameCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void registerPPCallbacks(CompilerInstance &Compiler) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  SourceRangeSet SuppressMacroExpansions;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_LAMBDAFUNCTIONNAMECHECK_H
