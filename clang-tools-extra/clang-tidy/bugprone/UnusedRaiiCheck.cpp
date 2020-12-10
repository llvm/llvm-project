//===--- UnusedRaiiCheck.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnusedRaiiCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

namespace {
AST_MATCHER(CXXRecordDecl, hasNonTrivialDestructor) {
  // TODO: If the dtor is there but empty we don't want to warn either.
  return Node.hasDefinition() && Node.hasNonTrivialDestructor();
}
} // namespace

void UnusedRaiiCheck::registerMatchers(MatchFinder *Finder) {
  // Look for temporaries that are constructed in-place and immediately
  // destroyed. Look for temporaries created by a functional cast but not for
  // those returned from a call.
  auto BindTemp = cxxBindTemporaryExpr(
                      unless(has(ignoringParenImpCasts(callExpr()))),
                      unless(has(ignoringParenImpCasts(objcMessageExpr()))))
                      .bind("temp");
  Finder->addMatcher(
      traverse(ast_type_traits::TK_AsIs,
               exprWithCleanups(
                   unless(isInTemplateInstantiation()),
                   hasParent(compoundStmt().bind("compound")),
                   hasType(cxxRecordDecl(hasNonTrivialDestructor())),
                   anyOf(has(ignoringParenImpCasts(BindTemp)),
                         has(ignoringParenImpCasts(cxxFunctionalCastExpr(
                             has(ignoringParenImpCasts(BindTemp)))))))
                   .bind("expr")),
      this);
}

void UnusedRaiiCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *E = Result.Nodes.getNodeAs<Expr>("expr");

  // We ignore code expanded from macros to reduce the number of false
  // positives.
  if (E->getBeginLoc().isMacroID())
    return;

  // Don't emit a warning for the last statement in the surrounding compound
  // statement.
  const auto *CS = Result.Nodes.getNodeAs<CompoundStmt>("compound");
  if (E == CS->body_back())
    return;

  // Emit a warning.
  auto D = diag(E->getBeginLoc(), "object destroyed immediately after "
                                  "creation; did you mean to name the object?");
  const char *Replacement = " give_me_a_name";

  // If this is a default ctor we have to remove the parens or we'll introduce a
  // most vexing parse.
  const auto *BTE = Result.Nodes.getNodeAs<CXXBindTemporaryExpr>("temp");
  if (const auto *TOE = dyn_cast<CXXTemporaryObjectExpr>(BTE->getSubExpr()))
    if (TOE->getNumArgs() == 0) {
      D << FixItHint::CreateReplacement(
          CharSourceRange::getTokenRange(TOE->getParenOrBraceRange()),
          Replacement);
      return;
    }

  // Otherwise just suggest adding a name. To find the place to insert the name
  // find the first TypeLoc in the children of E, which always points to the
  // written type.
  auto Matches =
      match(expr(hasDescendant(typeLoc().bind("t"))), *E, *Result.Context);
  if (const auto *TL = selectFirst<TypeLoc>("t", Matches))
    D << FixItHint::CreateInsertion(
        Lexer::getLocForEndOfToken(TL->getEndLoc(), 0, *Result.SourceManager,
                                   getLangOpts()),
        Replacement);
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
