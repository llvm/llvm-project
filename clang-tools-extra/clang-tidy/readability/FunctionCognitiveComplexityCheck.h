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
/// There is only one configuration option:
///
///   * `Threshold` - flag functions with Cognitive Complexity exceeding
///     this number. The default is `25`.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-function-cognitive-complexity.html
class FunctionCognitiveComplexityCheck : public ClangTidyCheck {
public:
  FunctionCognitiveComplexityCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const unsigned Threshold;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONCOGNITIVECOMPLEXITYCHECK_H
