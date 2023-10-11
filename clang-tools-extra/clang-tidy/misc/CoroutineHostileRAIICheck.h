//===--- CoroutineHostileRAIICheck.h - clang-tidy -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_COROUTINESHOSTILERAIICHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_COROUTINESHOSTILERAIICHECK_H

#include "../ClangTidyCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace clang::tidy::misc {

/// Check detects objects which should not to persist across suspension points
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc/coroutine-hostile-raii.html
class CoroutineHostileRAIICheck : public ClangTidyCheck {
public:
  CoroutineHostileRAIICheck(llvm::StringRef Name,
                                  ClangTidyContext *Context);

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus20;
  }

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void checkVarDecl(VarDecl *VD);
  // List of fully qualified types which should not persist across a suspension
  // point in a coroutine.
  std::vector<StringRef> RAIIDenyList;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_COROUTINESHOSTILERAIICHECK_H
