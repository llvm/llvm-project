//===--- DanglingHandleCheck.h - clang-tidy----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DANGLING_HANDLE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DANGLING_HANDLE_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Detect dangling references in value handlers like
/// std::experimental::string_view.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/dangling-handle.html
class DanglingHandleCheck : public ClangTidyCheck {
public:
  DanglingHandleCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  void registerMatchersForVariables(ast_matchers::MatchFinder *Finder);
  void registerMatchersForReturn(ast_matchers::MatchFinder *Finder);

  const std::vector<StringRef> HandleClasses;
  const ast_matchers::internal::Matcher<RecordDecl> IsAHandle;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_DANGLING_HANDLE_H
