//===--- MoveSharedPointerContentsCheck.h - clang-tidy ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MOVESHAREDPOINTERCONTENTSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MOVESHAREDPOINTERCONTENTSCHECK_H

#include <vector>

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Detects calls to move the contents out of a ``std::shared_ptr`` rather than
/// moving the pointer itself.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/move-shared-pointer-contents.html
class MoveSharedPointerContentsCheck : public ClangTidyCheck {
public:
  MoveSharedPointerContentsCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool
  isLanguageVersionSupported(const LangOptions &LangOptions) const override {
    return LangOptions.CPlusPlus11;
  }
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  const std::vector<StringRef> SharedPointerClasses;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MOVESHAREDPOINTERCONTENTSCHECK_H
