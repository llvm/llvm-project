//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantConstCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static std::optional<Token> findConstToRemove(const VarDecl *VD,
                                              const SourceManager &SM,
                                              const ASTContext &Context) {
  const SourceLocation NameBeginLoc = VD->getQualifier()
                                          ? VD->getQualifierLoc().getBeginLoc()
                                          : VD->getLocation();

  const bool IsPointer =
      VD->getType()->isPointerType() || VD->getType()->isMemberPointerType();

  // If the 'findPreviousTokenKind' below fails,
  // we know it is a pointer but cannot find the start token.
  // This can happen when either type is aliased or `auto` was used.
  // e.g: constexpr const auto const str = "hello";
  // In cases like this, Clang already warns about the use of const
  // as duplicate, so we can safely ignore these cases.
  const SourceLocation ConstSearchStartLoc =
      !IsPointer ? VD->getBeginLoc()
                 : utils::lexer::findPreviousTokenKind(
                       NameBeginLoc, SM, Context.getLangOpts(), tok::star);

  if (ConstSearchStartLoc.isInvalid())
    return std::nullopt;

  const SourceLocation PrevSemi = utils::lexer::findPreviousAnyTokenKind(
      NameBeginLoc, SM, Context.getLangOpts(), tok::semi);

  // Verify that there is no semicolon between ConstSearchStartLoc and
  // NameBeginLoc. This is to limit search area for our variable decl only
  if (PrevSemi.isValid() &&
      SM.isBeforeInTranslationUnit(ConstSearchStartLoc, PrevSemi))
    return std::nullopt;

  const CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getCharRange(ConstSearchStartLoc, NameBeginLoc), SM,
      Context.getLangOpts());

  if (FileRange.isInvalid())
    return std::nullopt;

  return utils::lexer::getQualifyingToken(tok::kw_const, FileRange, Context,
                                          SM);
}

RedundantConstCheck::RedundantConstCheck(StringRef Name,
                                         ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void RedundantConstCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      varDecl(isConstexpr(), unless(hasType(referenceType()))).bind("var_decl"),
      this);
}

static bool hasProblemSibling(const VarDecl *VD, SourceLocation ConstLoc,
                              const SourceManager &SM, const LangOptions &LO) {
  const SourceLocation Semi =
      utils::lexer::findNextAnyTokenKind(ConstLoc, SM, LO, tok::semi);
  if (Semi.isInvalid())
    return false;

  const SourceLocation Comma =
      utils::lexer::findNextAnyTokenKind(ConstLoc, SM, LO, tok::comma);
  if (Comma.isInvalid() || !SM.isBeforeInTranslationUnit(Comma, Semi))
    return false;

  for (Decl *D : VD->getDeclContext()->decls()) {
    if (SM.isBeforeInTranslationUnit(Semi, D->getBeginLoc()))
      break;

    const auto *Sib = dyn_cast<VarDecl>(D);
    if (!Sib || Sib == VD)
      continue;

    const SourceLocation N = Sib->getLocation();
    if (SM.isBeforeInTranslationUnit(N, ConstLoc) ||
        SM.isBeforeInTranslationUnit(Semi, N))
      continue;

    const QualType T = Sib->getType();
    if (T->isReferenceType() || T->isPointerType() || T->isMemberPointerType())
      return true;
  }
  return false;
}

void RedundantConstCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *VD = Result.Nodes.getNodeAs<VarDecl>("var_decl");

  // Since we cannot tell the difference between `constexpr const` and
  // `constexpr` from the AST only, if we cannot find the actual `const` token,
  // we cannot do anything.
  const std::optional<Token> Tok =
      findConstToRemove(VD, *Result.SourceManager, *Result.Context);
  if (!Tok)
    return;

  const SourceLocation ConstLoc = Tok->getLocation();
  auto Diag =
      diag(ConstLoc,
           "redundant use of 'const'; 'constexpr' already implies 'const'");
  if (!hasProblemSibling(VD, ConstLoc, *Result.SourceManager,
                         Result.Context->getLangOpts()))
    Diag << FixItHint::CreateRemoval(ConstLoc);
}

} // namespace clang::tidy::readability
