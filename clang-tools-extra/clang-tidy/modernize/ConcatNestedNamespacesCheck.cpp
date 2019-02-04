//===--- ConcatNestedNamespacesCheck.cpp - clang-tidy----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConcatNestedNamespacesCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <algorithm>
#include <iterator>

namespace clang {
namespace tidy {
namespace modernize {

static bool locationsInSameFile(const SourceManager &Sources,
                                SourceLocation Loc1, SourceLocation Loc2) {
  return Loc1.isFileID() && Loc2.isFileID() &&
         Sources.getFileID(Loc1) == Sources.getFileID(Loc2);
}

static bool anonymousOrInlineNamespace(const NamespaceDecl &ND) {
  return ND.isAnonymousNamespace() || ND.isInlineNamespace();
}

static bool singleNamedNamespaceChild(const NamespaceDecl &ND) {
  NamespaceDecl::decl_range Decls = ND.decls();
  if (std::distance(Decls.begin(), Decls.end()) != 1)
    return false;

  const auto *ChildNamespace = dyn_cast<const NamespaceDecl>(*Decls.begin());
  return ChildNamespace && !anonymousOrInlineNamespace(*ChildNamespace);
}

static bool alreadyConcatenated(std::size_t NumCandidates,
                                const SourceRange &ReplacementRange,
                                const SourceManager &Sources,
                                const LangOptions &LangOpts) {
  CharSourceRange TextRange =
      Lexer::getAsCharRange(ReplacementRange, Sources, LangOpts);
  StringRef CurrentNamespacesText =
      Lexer::getSourceText(TextRange, Sources, LangOpts);
  return CurrentNamespacesText.count(':') == (NumCandidates - 1) * 2;
}

ConcatNestedNamespacesCheck::NamespaceString
ConcatNestedNamespacesCheck::concatNamespaces() {
  NamespaceString Result("namespace ");
  Result.append(Namespaces.front()->getName());

  std::for_each(std::next(Namespaces.begin()), Namespaces.end(),
                [&Result](const NamespaceDecl *ND) {
                  Result.append("::");
                  Result.append(ND->getName());
                });

  return Result;
}

void ConcatNestedNamespacesCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus17)
    return;

  Finder->addMatcher(ast_matchers::namespaceDecl().bind("namespace"), this);
}

void ConcatNestedNamespacesCheck::reportDiagnostic(
    const SourceRange &FrontReplacement, const SourceRange &BackReplacement) {
  diag(Namespaces.front()->getBeginLoc(),
       "nested namespaces can be concatenated", DiagnosticIDs::Warning)
      << FixItHint::CreateReplacement(FrontReplacement, concatNamespaces())
      << FixItHint::CreateReplacement(BackReplacement, "}");
}

void ConcatNestedNamespacesCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const NamespaceDecl &ND = *Result.Nodes.getNodeAs<NamespaceDecl>("namespace");
  const SourceManager &Sources = *Result.SourceManager;

  if (!locationsInSameFile(Sources, ND.getBeginLoc(), ND.getRBraceLoc()))
    return;

  if (!Sources.isInMainFile(ND.getBeginLoc()))
    return;

  if (anonymousOrInlineNamespace(ND))
    return;

  Namespaces.push_back(&ND);

  if (singleNamedNamespaceChild(ND))
    return;

  SourceRange FrontReplacement(Namespaces.front()->getBeginLoc(),
                               Namespaces.back()->getLocation());
  SourceRange BackReplacement(Namespaces.back()->getRBraceLoc(),
                              Namespaces.front()->getRBraceLoc());

  if (!alreadyConcatenated(Namespaces.size(), FrontReplacement, Sources,
                           getLangOpts()))
    reportDiagnostic(FrontReplacement, BackReplacement);

  Namespaces.clear();
}

} // namespace modernize
} // namespace tidy
} // namespace clang
