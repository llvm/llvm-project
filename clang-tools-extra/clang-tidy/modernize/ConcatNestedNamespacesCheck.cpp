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

std::optional<SourceRange>
NS::getCleanedNamespaceFrontRange(const SourceManager &SM,
                                  const LangOptions &LangOpts) const {
  // Front from namespace tp '{'
  std::optional<Token> Tok =
      ::clang::tidy::utils::lexer::findNextTokenSkippingComments(
          back()->getLocation(), SM, LangOpts);
  if (!Tok)
    return std::nullopt;
  while (Tok->getKind() != tok::TokenKind::l_brace) {
    Tok = utils::lexer::findNextTokenSkippingComments(Tok->getEndLoc(), SM,
                                                      LangOpts);
    if (!Tok)
      return std::nullopt;
  }
  return SourceRange{front()->getBeginLoc(), Tok->getEndLoc()};
}
SourceRange NS::getReplacedNamespaceFrontRange() const {
  return SourceRange{front()->getBeginLoc(), back()->getLocation()};
}

SourceRange NS::getDefaultNamespaceBackRange() const {
  return SourceRange{front()->getRBraceLoc(), front()->getRBraceLoc()};
}
SourceRange NS::getNamespaceBackRange(const SourceManager &SM,
                                      const LangOptions &LangOpts) const {
  // Back from '}' to conditional '// namespace xxx'
  SourceLocation Loc = front()->getRBraceLoc();
  std::optional<Token> Tok =
      utils::lexer::findNextTokenIncludingComments(Loc, SM, LangOpts);
  if (!Tok)
    return getDefaultNamespaceBackRange();
  if (Tok->getKind() != tok::TokenKind::comment)
    return getDefaultNamespaceBackRange();
  SourceRange TokRange = SourceRange{Tok->getLocation(), Tok->getEndLoc()};
  StringRef TokText = getRawStringRef(TokRange, SM, LangOpts);
  NamespaceName CloseComment{"namespace "};
  appendCloseComment(CloseComment);
  // current fix hint in readability/NamespaceCommentCheck.cpp use single line
  // comment
  constexpr size_t L = sizeof("//") - 1U;
  if (TokText.take_front(L) == "//" &&
      TokText.drop_front(L).trim() != CloseComment)
    return getDefaultNamespaceBackRange();
  return SourceRange{front()->getRBraceLoc(), Tok->getEndLoc()};
}

void NS::appendName(NamespaceName &Str) const {
  for (const NamespaceDecl *ND : *this) {
    if (ND->isInlineNamespace())
      Str.append("inline ");
    Str.append(ND->getName());
    if (ND != back())
      Str.append("::");
  }
}
void NS::appendCloseComment(NamespaceName &Str) const {
  if (size() == 1)
    Str.append(back()->getName());
  else
    appendName(Str);
}

bool ConcatNestedNamespacesCheck::unsupportedNamespace(const NamespaceDecl &ND,
                                                       bool IsChild) const {
  if (ND.isAnonymousNamespace() || !ND.attrs().empty())
    return true;
  if (getLangOpts().CPlusPlus20) {
    // C++20 support inline nested namespace
    bool IsFirstNS = IsChild || !Namespaces.empty();
    return ND.isInlineNamespace() && !IsFirstNS;
  }
  return ND.isInlineNamespace();
}

bool ConcatNestedNamespacesCheck::singleNamedNamespaceChild(
    const NamespaceDecl &ND) const {
  NamespaceDecl::decl_range Decls = ND.decls();
  if (std::distance(Decls.begin(), Decls.end()) != 1)
    return false;

  const auto *ChildNamespace = dyn_cast<const NamespaceDecl>(*Decls.begin());
  return ChildNamespace && !unsupportedNamespace(*ChildNamespace, true);
}

void ConcatNestedNamespacesCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(ast_matchers::namespaceDecl().bind("namespace"), this);
}

void ConcatNestedNamespacesCheck::reportDiagnostic(
    const SourceManager &SM, const LangOptions &LangOpts) {
  DiagnosticBuilder DB =
      diag(Namespaces.front().front()->getBeginLoc(),
           "nested namespaces can be concatenated", DiagnosticIDs::Warning);

  SmallVector<SourceRange, 6> Fronts;
  Fronts.reserve(Namespaces.size() - 1U);
  SmallVector<SourceRange, 6> Backs;
  Backs.reserve(Namespaces.size());

  for (const NS &ND : Namespaces) {
    std::optional<SourceRange> SR =
        ND.getCleanedNamespaceFrontRange(SM, LangOpts);
    if (!SR)
      return;
    Fronts.push_back(SR.value());
    Backs.push_back(ND.getNamespaceBackRange(SM, LangOpts));
  }
  if (Fronts.empty() || Backs.empty())
    return;

  // the last one should be handled specially
  Fronts.pop_back();
  SourceRange LastRBrace = Backs.pop_back_val();

  NamespaceName ConcatNameSpace{"namespace "};
  for (const NS &NS : Namespaces) {
    NS.appendName(ConcatNameSpace);
    if (&NS != &Namespaces.back()) // compare address directly
      ConcatNameSpace.append("::");
  }

  for (SourceRange const &Front : Fronts)
    DB << FixItHint::CreateRemoval(Front);
  DB << FixItHint::CreateReplacement(
      Namespaces.back().getReplacedNamespaceFrontRange(), ConcatNameSpace);
  if (LastRBrace != Namespaces.back().getDefaultNamespaceBackRange())
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

  if (unsupportedNamespace(ND, false))
    return;

  if (!ND.isNested())
    Namespaces.push_back(NS{});
  if (!Namespaces.empty())
    // Otherwise it will crash with invalid input like `inline namespace
    // a::b::c`.
    Namespaces.back().push_back(&ND);

  if (singleNamedNamespaceChild(ND))
    return;

  if (Namespaces.size() > 1)
    reportDiagnostic(Sources, getLangOpts());

  Namespaces.clear();
}

} // namespace clang::tidy::modernize
