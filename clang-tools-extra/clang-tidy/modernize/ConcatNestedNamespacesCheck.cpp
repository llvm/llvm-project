//===--- ConcatNestedNamespacesCheck.cpp - clang-tidy----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConcatNestedNamespacesCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <optional>

namespace clang::tidy::modernize {

static bool locationsInSameFile(const SourceManager &Sources,
                                SourceLocation Loc1, SourceLocation Loc2) {
  return Loc1.isFileID() && Loc2.isFileID() &&
         Sources.getFileID(Loc1) == Sources.getFileID(Loc2);
}

static StringRef getRawStringRef(const SourceRange &Range,
                                 const SourceManager &Sources,
                                 const LangOptions &LangOpts) {
  CharSourceRange TextRange = Lexer::getAsCharRange(Range, Sources, LangOpts);
  return Lexer::getSourceText(TextRange, Sources, LangOpts);
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
  // FIXME: This logic breaks when there is a comment with ':'s in the middle.
  return getRawStringRef(ReplacementRange, Sources, LangOpts).count(':') ==
         (NumCandidates - 1) * 2;
}

static std::optional<SourceRange>
getCleanedNamespaceFrontRange(const NamespaceDecl *ND, const SourceManager &SM,
                              const LangOptions &LangOpts) {
  // Front from namespace tp '{'
  std::optional<Token> Tok =
      ::clang::tidy::utils::lexer::findNextTokenSkippingComments(
          ND->getLocation(), SM, LangOpts);
  if (!Tok)
    return std::nullopt;
  while (Tok->getKind() != tok::TokenKind::l_brace) {
    Tok = utils::lexer::findNextTokenSkippingComments(Tok->getEndLoc(), SM,
                                                      LangOpts);
    if (!Tok)
      return std::nullopt;
  }
  return SourceRange{ND->getBeginLoc(), Tok->getEndLoc()};
}

static SourceRange getCleanedNamespaceBackRange(const NamespaceDecl *ND,
                                                const SourceManager &SM,
                                                const LangOptions &LangOpts) {
  // Back from '}' to conditional '// namespace xxx'
  const SourceRange DefaultSourceRange =
      SourceRange{ND->getRBraceLoc(), ND->getRBraceLoc()};
  SourceLocation Loc = ND->getRBraceLoc();
  std::optional<Token> Tok =
      utils::lexer::findNextTokenIncludingComments(Loc, SM, LangOpts);
  if (!Tok)
    return DefaultSourceRange;
  if (Tok->getKind() != tok::TokenKind::comment)
    return DefaultSourceRange;
  SourceRange TokRange = SourceRange{Tok->getLocation(), Tok->getEndLoc()};
  StringRef TokText = getRawStringRef(TokRange, SM, LangOpts);
  std::string CloseComment = "namespace " + ND->getNameAsString();
  // current fix hint in readability/NamespaceCommentCheck.cpp use single line
  // comment
  if (TokText != "// " + CloseComment && TokText != "//" + CloseComment)
    return DefaultSourceRange;
  return SourceRange{ND->getRBraceLoc(), Tok->getEndLoc()};
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
  Finder->addMatcher(ast_matchers::namespaceDecl().bind("namespace"), this);
}

void ConcatNestedNamespacesCheck::reportDiagnostic(
    const SourceManager &SM, const LangOptions &LangOpts) {
  DiagnosticBuilder DB =
      diag(Namespaces.front()->getBeginLoc(),
           "nested namespaces can be concatenated", DiagnosticIDs::Warning);

  SmallVector<SourceRange, 6> Fronts;
  Fronts.reserve(Namespaces.size() - 1U);
  SmallVector<SourceRange, 6> Backs;
  Backs.reserve(Namespaces.size());

  NamespaceDecl const *LastND = nullptr;

  for (const NamespaceDecl *ND : Namespaces) {
    if (ND->isNested())
      continue;
    LastND = ND;
    std::optional<SourceRange> SR =
        getCleanedNamespaceFrontRange(ND, SM, LangOpts);
    if (!SR.has_value())
      return;
    Fronts.push_back(SR.value());
    Backs.push_back(getCleanedNamespaceBackRange(ND, SM, LangOpts));
  }
  if (LastND == nullptr || Fronts.empty() || Backs.empty())
    return;
  // the last one should be handled specially
  Fronts.pop_back();
  SourceRange LastRBrace = Backs.pop_back_val();
  NamespaceString ConcatNameSpace = concatNamespaces();

  for (SourceRange const &Front : Fronts)
    DB << FixItHint::CreateRemoval(Front);
  DB << FixItHint::CreateReplacement(
      SourceRange{LastND->getBeginLoc(), LastND->getLocation()},
      ConcatNameSpace);
  if (LastRBrace != SourceRange{LastND->getRBraceLoc(), LastND->getRBraceLoc()})
    DB << FixItHint::CreateReplacement(LastRBrace,
                                       ("} // " + ConcatNameSpace).str());
  for (SourceRange const &Back : llvm::reverse(Backs))
    DB << FixItHint::CreateRemoval(Back);
}

void ConcatNestedNamespacesCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const NamespaceDecl &ND = *Result.Nodes.getNodeAs<NamespaceDecl>("namespace");
  const SourceManager &Sources = *Result.SourceManager;

  if (!locationsInSameFile(Sources, ND.getBeginLoc(), ND.getRBraceLoc()))
    return;

  if (anonymousOrInlineNamespace(ND))
    return;

  Namespaces.push_back(&ND);

  if (singleNamedNamespaceChild(ND))
    return;

  SourceRange FrontReplacement(Namespaces.front()->getBeginLoc(),
                               Namespaces.back()->getLocation());

  if (!alreadyConcatenated(Namespaces.size(), FrontReplacement, Sources,
                           getLangOpts()))
    reportDiagnostic(Sources, getLangOpts());

  Namespaces.clear();
}

} // namespace clang::tidy::modernize
