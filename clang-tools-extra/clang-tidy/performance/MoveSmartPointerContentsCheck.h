//===--- MoveSmartPointerContentsCheck.h - clang-tidy -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_MOVESMARTPOINTERCONTENTSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_MOVESMARTPOINTERCONTENTSCHECK_H

#include <vector>

#include "../ClangTidyCheck.h"

namespace clang::tidy::performance {

/// Checks that std::move is not called on the contents of a smart pointer and
/// suggests moving out of the pointer instead.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance/move-smart-pointer-contents.html
class MoveSmartPointerContentsCheck : public ClangTidyCheck {
public:
  MoveSmartPointerContentsCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const std::vector<StringRef> UniquePointerClasses;
  const ast_matchers::internal::Matcher<RecordDecl> IsAUniquePointer;
  const std::vector<StringRef> SharedPointerClasses;
  const ast_matchers::internal::Matcher<RecordDecl> IsASharedPointer;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_MOVESMARTPOINTERCONTENTSCHECK_H
