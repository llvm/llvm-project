//===--- NamespaceCommentCheck.h - clang-tidy -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NAMESPACECOMMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NAMESPACECOMMENTCHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/Support/Regex.h"

namespace clang {
namespace tidy {
namespace readability {

/// Checks that long namespaces have a closing comment.
///
/// http://llvm.org/docs/CodingStandards.html#namespace-indentation
///
/// https://google.github.io/styleguide/cppguide.html#Namespaces
class NamespaceCommentCheck : public ClangTidyCheck {
public:
  NamespaceCommentCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void storeOptions(ClangTidyOptions::OptionMap &Options) override;

  llvm::Regex NamespaceCommentPattern;
  const unsigned ShortNamespaceLines;
  const unsigned SpacesBeforeComments;
  llvm::SmallVector<SourceLocation, 4> Ends;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NAMESPACECOMMENTCHECK_H
