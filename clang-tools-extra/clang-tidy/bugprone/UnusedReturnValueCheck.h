//===--- UnusedReturnValueCheck.h - clang-tidy-------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNUSEDRETURNVALUECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNUSEDRETURNVALUECHECK_H

#include "../ClangTidyCheck.h"
#include <string>

namespace clang::tidy::bugprone {

/// Detects function calls where the return value is unused.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/unused-return-value.html
class UnusedReturnValueCheck : public ClangTidyCheck {
public:
  UnusedReturnValueCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  const std::vector<StringRef> CheckedFunctions;
  const std::vector<StringRef> CheckedReturnTypes;

protected:
  UnusedReturnValueCheck(StringRef Name, ClangTidyContext *Context,
                         std::vector<StringRef> CheckedFunctions);
  UnusedReturnValueCheck(StringRef Name, ClangTidyContext *Context,
                         std::vector<StringRef> CheckedFunctions,
                         std::vector<StringRef> CheckedReturnTypes,
                         bool AllowCastToVoid);
  bool AllowCastToVoid;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_UNUSEDRETURNVALUECHECK_H
