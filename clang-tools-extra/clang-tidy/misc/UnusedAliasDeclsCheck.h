//===--- UnusedAliasDeclsCheck.h - clang-tidy--------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_ALIAS_DECLS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_ALIAS_DECLS_H

#include "../ClangTidy.h"
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
