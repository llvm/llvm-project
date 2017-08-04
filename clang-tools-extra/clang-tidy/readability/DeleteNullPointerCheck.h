//===--- DeleteNullPointerCheck.h - clang-tidy-------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_DELETE_NULL_POINTER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_DELETE_NULL_POINTER_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// Check whether the 'if' statement is unnecessary before calling 'delete' on a
/// pointer.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-delete-null-pointer.html
class DeleteNullPointerCheck : public ClangTidyCheck {
public:
  DeleteNullPointerCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_DELETE_NULL_POINTER_H
