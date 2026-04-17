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

static std::optional<Token>
findConstToRemove(const VarDecl *VD, const MatchFinder::MatchResult &Result) {
  const SourceManager &SM = *Result.SourceManager;

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
      IsPointer
          ? utils::lexer::findPreviousTokenKind(
                NameBeginLoc, SM, Result.Context->getLangOpts(), tok::star)
          : VD->getBeginLoc();

  if (ConstSearchStartLoc.isInvalid())
    return std::nullopt;

  const SourceLocation PrevSemi = utils::lexer::findPreviousAnyTokenKind(
      NameBeginLoc, SM, Result.Context->getLangOpts(), tok::semi);

  // Verify that there is no semicolon between ConstSearchStartLoc and
  // NameBeginLoc. This is to limit search area for our variable decl only
  if (!PrevSemi.isInvalid() &&
      SM.isBeforeInTranslationUnit(ConstSearchStartLoc, PrevSemi))
    return std::nullopt;

  const CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getCharRange(ConstSearchStartLoc, NameBeginLoc), SM,
      Result.Context->getLangOpts());

  if (FileRange.isInvalid())
    return std::nullopt;

  return utils::lexer::getQualifyingToken(tok::kw_const, FileRange,
                                          *Result.Context, SM);
}

RedundantConstCheck::RedundantConstCheck(StringRef Name,
                                         ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void RedundantConstCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      varDecl(isConstexpr(), unless(hasType(referenceType()))).bind("var_decl"),
      this);
}

void RedundantConstCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *VD = Result.Nodes.getNodeAs<VarDecl>("var_decl");

  // Since we cannot tell the difference between `constexpr const` and
  // `constexpr` from the AST only, if we cannot find the actual `const` token,
  // we cannot do anything
  const std::optional<Token> Tok = findConstToRemove(VD, Result);
  if (!Tok)
    return;

  diag(Tok->getLocation(),
       "redundant use of 'const'; 'constexpr' already implies 'const'")
      << FixItHint::CreateRemoval(Tok->getLocation());
}

bool RedundantConstCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus11 || LangOpts.C23;
}

std::optional<TraversalKind>
RedundantConstCheck::getCheckTraversalKind() const {
  return TK_IgnoreUnlessSpelledInSource;
}

} // namespace clang::tidy::readability
