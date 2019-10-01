//===--- UnusedAliasDeclsCheck.h - clang-tidy--------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_ALIAS_DECLS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_ALIAS_DECLS_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
namespace tidy {
namespace misc {

/// Finds unused namespace alias declarations.
class UnusedAliasDeclsCheck : public ClangTidyCheck {
public:
  UnusedAliasDeclsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;

private:
  llvm::DenseMap<const NamedDecl *, CharSourceRange> FoundDecls;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_ALIAS_DECLS_H
