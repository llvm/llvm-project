//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SCOPEREDUCTIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SCOPEREDUCTIONCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::misc {

class ScopeReductionCheck : public ClangTidyCheck {
public:
  ScopeReductionCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void emitUsageNotes(const llvm::SmallVector<const DeclRefExpr *, 8> &Uses);

  template <typename Container>
  auto take(const Container &ThisContainer, size_t n) {
    return llvm::make_range(ThisContainer.begin(),
                            ThisContainer.begin() +
                                std::min(n, ThisContainer.size()));
  }
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_SCOPEREDUCTIONCHECK_H
