//===--- FunctionCognitiveComplexityCheck.h - clang-tidy --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONCOGNITIVECOMPLEXITYCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONCOGNITIVECOMPLEXITYCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace readability {

/// Checks function Cognitive Complexity metric.
///
/// There are the following configuration option:
///
///   * `Threshold` - flag functions with Cognitive Complexity exceeding
///     this number. The default is `25`.
///   * `DescribeBasicIncrements`- if set to `true`, then for each function
///     exceeding the complexity threshold the check will issue additional
///     diagnostics on every piece of code (loop, `if` statement, etc.) which
///     contributes to that complexity.
//      Default is `true`
///   * `IgnoreMacros` - if set to `true`, the check will ignore code inside
///     macros. Default is `false`.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-function-cognitive-complexity.html
class FunctionCognitiveComplexityCheck : public ClangTidyCheck {
public:
  FunctionCognitiveComplexityCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  llvm::Optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  const unsigned Threshold;
  const bool DescribeBasicIncrements;
  const bool IgnoreMacros;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONCOGNITIVECOMPLEXITYCHECK_H
