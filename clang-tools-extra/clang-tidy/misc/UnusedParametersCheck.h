//===--- UnusedParametersCheck.h - clang-tidy--------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_PARAMETERS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_PARAMETERS_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::misc {

/// Finds unused parameters and fixes them, so that `-Wunused-parameter` can be
/// turned on.
class UnusedParametersCheck : public ClangTidyCheck {
public:
  UnusedParametersCheck(StringRef Name, ClangTidyContext *Context);
  ~UnusedParametersCheck();
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const bool StrictMode;
  class IndexerVisitor;
  std::unique_ptr<IndexerVisitor> Indexer;

  void
  warnOnUnusedParameter(const ast_matchers::MatchFinder::MatchResult &Result,
                        const FunctionDecl *Function, unsigned ParamIndex);
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_PARAMETERS_H
