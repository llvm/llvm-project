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
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;

  void addMacro(const std::string &Name, const std::string &Value) noexcept;

private:
  void storeOptions(ClangTidyOptions::OptionMap &Options) override;
  std::string getNamespaceComment(const NamespaceDecl *ND,
                                  bool InsertLineBreak);
  std::string getNamespaceComment(const std::string &NameSpaceName,
                                  bool InsertLineBreak);
  bool isNamespaceMacroDefinition(const StringRef NameSpaceName);
  std::tuple<bool, StringRef>
  isNamespaceMacroExpansion(const StringRef NameSpaceName);

  llvm::Regex NamespaceCommentPattern;
  const unsigned ShortNamespaceLines;
  const unsigned SpacesBeforeComments;
  llvm::SmallVector<SourceLocation, 4> Ends;

  // Store macros to verify that warning is not thrown when namespace name is a
  // preprocessed define.
  std::vector<std::pair<std::string, std::string>> Macros;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_NAMESPACECOMMENTCHECK_H
