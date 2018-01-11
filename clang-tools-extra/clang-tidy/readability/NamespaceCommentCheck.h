//===--- NamespaceCommentCheck.h - clang-tidy -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NAMESPACECOMMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NAMESPACECOMMENTCHECK_H

#include "../ClangTidy.h"
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
