//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTOVERRIDECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTOVERRIDECHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace clang::tidy::readability {

/// Finds overrides that only forward to the inherited implementation without
/// changing its interface.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/readability/redundant-override.html
class RedundantOverrideCheck : public ClangTidyCheck {
public:
  using ClangTidyCheck::ClangTidyCheck;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;

private:
  std::optional<TraversalKind> getCheckTraversalKind() const override;

  SmallVector<const CXXMethodDecl *, 8> Candidates;
  llvm::SmallPtrSet<const CXXMethodDecl *, 8> ReferencedMethods;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTOVERRIDECHECK_H
