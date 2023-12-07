//===--- ElseAfterReturnCheck.h - clang-tidy---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_ELSEAFTERRETURNCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_ELSEAFTERRETURNCHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/DenseMap.h"

namespace clang::tidy::readability {

/// Flags the usages of `else` after `return`.
///
/// http://llvm.org/docs/CodingStandards.html#don-t-use-else-after-a-return
class ElseAfterReturnCheck : public ClangTidyCheck {
public:
  ElseAfterReturnCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

  using ConditionalBranchMap =
      llvm::DenseMap<FileID, SmallVector<SourceRange, 1>>;

private:
  const bool WarnOnUnfixable;
  const bool WarnOnConditionVariables;
  ConditionalBranchMap PPConditionals;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_ELSEAFTERRETURNCHECK_H
