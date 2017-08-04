//===--- UnusedParametersCheck.h - clang-tidy--------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_PARAMETERS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_PARAMETERS_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// Finds unused parameters and fixes them, so that `-Wunused-parameter` can be
/// turned on.
class UnusedParametersCheck : public ClangTidyCheck {
public:
  UnusedParametersCheck(StringRef Name, ClangTidyContext *Context);
  ~UnusedParametersCheck();
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  class IndexerVisitor;
  std::unique_ptr<IndexerVisitor> Indexer;

  void
  warnOnUnusedParameter(const ast_matchers::MatchFinder::MatchResult &Result,
                        const FunctionDecl *Function, unsigned ParamIndex);
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_PARAMETERS_H
