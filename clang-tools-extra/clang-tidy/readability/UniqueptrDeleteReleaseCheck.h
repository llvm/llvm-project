//===--- UniqueptrDeleteReleaseCheck.h - clang-tidy--------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_UNIQUEPTR_DELETE_RELEASE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_UNIQUEPTR_DELETE_RELEASE_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// Flags statements of the form ``delete <unique_ptr expr>.release();`` and
/// replaces them with: ``<unique_ptr expr> = nullptr;``
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-uniqueptr-delete-release.html
class UniqueptrDeleteReleaseCheck : public ClangTidyCheck {
public:
  UniqueptrDeleteReleaseCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_UNIQUEPTR_DELETE_RELEASE_H
