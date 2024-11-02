//===--- ConfusableIdentifierCheck.h - clang-tidy
//-------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_CONFUSABLE_IDENTIFIER_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_CONFUSABLE_IDENTIFIER_CHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace misc {

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

private:
  std::string skeleton(StringRef);
  llvm::StringMap<llvm::SmallVector<const NamedDecl *>> Mapper;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_CONFUSABLE_IDENTIFIER_CHECK_H
