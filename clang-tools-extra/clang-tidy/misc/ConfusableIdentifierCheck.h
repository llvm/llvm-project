//===--- ConfusableIdentifierCheck.h - clang-tidy ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_CONFUSABLE_IDENTIFIER_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_CONFUSABLE_IDENTIFIER_CHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/DenseMap.h"

namespace clang::tidy::misc {

/// Finds symbol which have confusable identifiers, i.e. identifiers that look
/// the same visually but have a different Unicode representation.
/// If symbols are confusable but don't live in conflicting namespaces, they are
/// not reported.
class ConfusableIdentifierCheck : public ClangTidyCheck {
public:
  ConfusableIdentifierCheck(StringRef Name, ClangTidyContext *Context);
  ~ConfusableIdentifierCheck();

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  void addDeclToCheck(const NamedDecl *ND, const Decl *Parent);

  llvm::DenseMap<
      const IdentifierInfo *,
      llvm::SmallVector<std::pair<const NamedDecl *, const Decl *>, 1>>
      NameToDecls;
};

} // namespace clang::tidy::misc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_CONFUSABLE_IDENTIFIER_CHECK_H
