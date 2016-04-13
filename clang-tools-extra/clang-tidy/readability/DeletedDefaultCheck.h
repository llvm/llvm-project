//===--- DeletedDefaultCheck.h - clang-tidy----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_DELETED_DEFAULT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_DELETED_DEFAULT_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// Checks when a constructor or an assignment operator is marked as '= default'
/// but is actually deleted by the compiler.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-deleted-default.html
class DeletedDefaultCheck : public ClangTidyCheck {
public:
  DeletedDefaultCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_DELETED_DEFAULT_H
