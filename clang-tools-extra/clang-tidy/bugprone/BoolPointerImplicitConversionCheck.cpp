//===--- BoolPointerImplicitConversionCheck.cpp - clang-tidy --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BoolPointerImplicitConversionCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

void BoolPointerImplicitConversionCheck::registerMatchers(MatchFinder *Finder) {
  // Look for ifs that have an implicit bool* to bool conversion in the
  // condition. Filter negations.
  Finder->addMatcher(
      traverse(
          ast_type_traits::TK_AsIs,
          ifStmt(hasCondition(findAll(implicitCastExpr(
                     unless(hasParent(unaryOperator(hasOperatorName("!")))),
                     hasSourceExpression(expr(
                         hasType(pointerType(pointee(booleanType()))),
                         ignoringParenImpCasts(declRefExpr().bind("expr")))),
                     hasCastKind(CK_PointerToBoolean)))),
                 unless(isInTemplateInstantiation()))
              .bind("if")),
      this);
}

void BoolPointerImplicitConversionCheck::check(
    const MatchFinder::MatchResult &Result) {
  auto *If = Result.Nodes.getNodeAs<IfStmt>("if");
  auto *Var = Result.Nodes.getNodeAs<DeclRefExpr>("expr");

  // Ignore macros.
  if (Var->getBeginLoc().isMacroID())
    return;

  // Only allow variable accesses for now, no function calls or member exprs.
  // Check that we don't dereference the variable anywhere within the if. This
  // avoids false positives for checks of the pointer for nullptr before it is
  // dereferenced. If there is a dereferencing operator on this variable don't
  // emit a diagnostic. Also ignore array subscripts.
  const Decl *D = Var->getDecl();
  auto DeclRef = ignoringParenImpCasts(declRefExpr(to(equalsNode(D))));
  if (!match(findAll(
                 unaryOperator(hasOperatorName("*"), hasUnaryOperand(DeclRef))),
             *If, *Result.Context)
           .empty() ||
      !match(findAll(arraySubscriptExpr(hasBase(DeclRef))), *If,
             *Result.Context)
           .empty() ||
      // FIXME: We should still warn if the paremater is implicitly converted to
      // bool.
      !match(findAll(callExpr(hasAnyArgument(ignoringParenImpCasts(DeclRef)))),
             *If, *Result.Context)
           .empty() ||
      !match(findAll(cxxDeleteExpr(has(ignoringParenImpCasts(expr(DeclRef))))),
             *If, *Result.Context)
           .empty())
    return;

  diag(Var->getBeginLoc(), "dubious check of 'bool *' against 'nullptr', did "
                           "you mean to dereference it?")
      << FixItHint::CreateInsertion(Var->getBeginLoc(), "*");
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
