//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BoolPointerImplicitConversionCheck.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void BoolPointerImplicitConversionCheck::registerMatchers(MatchFinder *Finder) {
  // Look for ifs that have an implicit bool* to bool conversion in the
  // condition. Filter negations.
  Finder->addMatcher(
      traverse(
          TK_AsIs,
          ifStmt(
              hasCondition(findAll(implicitCastExpr(
                  unless(hasParent(unaryOperator(hasOperatorName("!")))),
                  hasSourceExpression(expr(
                      hasType(pointerType(pointee(booleanType()))),
                      ignoringParenImpCasts(anyOf(declRefExpr().bind("expr"),
                                                  memberExpr().bind("expr"))))),
                  hasCastKind(CK_PointerToBoolean)))),
              unless(isInTemplateInstantiation()))
              .bind("if")),
      this);
}

static void checkImpl(const MatchFinder::MatchResult &Result, const Expr *Ref,
                      const IfStmt *If,
                      const ast_matchers::internal::Matcher<Expr> &RefMatcher,
                      ClangTidyCheck &Check) {
  // Ignore macros.
  if (Ref->getBeginLoc().isMacroID())
    return;

  // Only allow variable accesses and member exprs for now, no function calls.
  // Check that we don't dereference the variable anywhere within the if. This
  // avoids false positives for checks of the pointer for nullptr before it is
  // dereferenced. If there is a dereferencing operator on this variable don't
  // emit a diagnostic. Also ignore array subscripts.
  if (!match(findAll(unaryOperator(hasOperatorName("*"),
                                   hasUnaryOperand(RefMatcher))),
             *If, *Result.Context)
           .empty() ||
      !match(findAll(arraySubscriptExpr(hasBase(RefMatcher))), *If,
             *Result.Context)
           .empty() ||
      // FIXME: We should still warn if the paremater is implicitly converted to
      // bool.
      !match(
           findAll(callExpr(hasAnyArgument(ignoringParenImpCasts(RefMatcher)))),
           *If, *Result.Context)
           .empty() ||
      !match(
           findAll(cxxDeleteExpr(has(ignoringParenImpCasts(expr(RefMatcher))))),
           *If, *Result.Context)
           .empty())
    return;

  Check.diag(Ref->getBeginLoc(),
             "dubious check of 'bool *' against 'nullptr', did "
             "you mean to dereference it?")
      << FixItHint::CreateInsertion(Ref->getBeginLoc(), "*");
}

void BoolPointerImplicitConversionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *If = Result.Nodes.getNodeAs<IfStmt>("if");
  if (const auto *E = Result.Nodes.getNodeAs<Expr>("expr")) {
    const Decl *D = isa<DeclRefExpr>(E) ? cast<DeclRefExpr>(E)->getDecl()
                                        : cast<MemberExpr>(E)->getMemberDecl();
    const auto M =
        ignoringParenImpCasts(anyOf(declRefExpr(to(equalsNode(D))),
                                    memberExpr(hasDeclaration(equalsNode(D)))));
    checkImpl(Result, E, If, M, *this);
  }
}

} // namespace clang::tidy::bugprone
