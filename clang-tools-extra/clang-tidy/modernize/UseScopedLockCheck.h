//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USESCOPEDLOCKCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USESCOPEDLOCKCHECK_H

#include "../ClangTidyCheck.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Stmt.h"
#include <optional>

namespace clang::tidy::modernize {

/// Finds uses of ``std::lock_guard`` and suggests replacing them with C++17's
/// alternative ``std::scoped_lock``.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/use-scoped-lock.html
class UseScopedLockCheck : public ClangTidyCheck {
public:
  UseScopedLockCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus17;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  void diagOnSingleLock(const VarDecl *LockGuard,
                        const ast_matchers::MatchFinder::MatchResult &Result);
  void diagOnMultipleLocks(
      const llvm::SmallVector<llvm::SmallVector<const VarDecl *>> &LockGroups,
      const ast_matchers::MatchFinder::MatchResult &Result);
  void diagOnSourceInfo(const TypeSourceInfo *LockGuardSourceInfo,
                        const ast_matchers::MatchFinder::MatchResult &Result);
  void diagOnUsingDecl(const UsingDecl *UsingDecl,
                       const ast_matchers::MatchFinder::MatchResult &Result);

  const bool WarnOnSingleLocks;
  const bool WarnOnUsingAndTypedef;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USESCOPEDLOCKCHECK_H
