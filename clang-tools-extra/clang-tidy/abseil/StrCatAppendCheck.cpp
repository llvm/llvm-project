//===--- StrCatAppendCheck.cpp - clang-tidy--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StrCatAppendCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

namespace {
// Skips any combination of temporary materialization, temporary binding and
// implicit casting.
AST_MATCHER_P(Stmt, IgnoringTemporaries, ast_matchers::internal::Matcher<Stmt>,
              InnerMatcher) {
  const Stmt *E = &Node;
  while (true) {
    if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(E))
      E = MTE->getSubExpr();
    if (const auto *BTE = dyn_cast<CXXBindTemporaryExpr>(E))
      E = BTE->getSubExpr();
    if (const auto *ICE = dyn_cast<ImplicitCastExpr>(E))
      E = ICE->getSubExpr();
    else
      break;
  }

  return InnerMatcher.matches(*E, Finder, Builder);
}

}  // namespace

// TODO: str += StrCat(...)
//       str.append(StrCat(...))

void StrCatAppendCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
  	return;
  const auto StrCat = functionDecl(hasName("::absl::StrCat"));
  // The arguments of absl::StrCat are implicitly converted to AlphaNum. This 
  // matches to the arguments because of that behavior. 
  const auto AlphaNum = IgnoringTemporaries(cxxConstructExpr(
      argumentCountIs(1), hasType(cxxRecordDecl(hasName("::absl::AlphaNum"))),
      hasArgument(0, ignoringImpCasts(declRefExpr(to(equalsBoundNode("LHS")),
                                                  expr().bind("Arg0"))))));

  const auto HasAnotherReferenceToLhs =
      callExpr(hasAnyArgument(expr(hasDescendant(declRefExpr(
          to(equalsBoundNode("LHS")), unless(equalsBoundNode("Arg0")))))));

  // Now look for calls to operator= with an object on the LHS and a call to
  // StrCat on the RHS. The first argument of the StrCat call should be the same
  // as the LHS. Ignore calls from template instantiations.
  Finder->addMatcher(
      cxxOperatorCallExpr(
          unless(isInTemplateInstantiation()), hasOverloadedOperatorName("="),
          hasArgument(0, declRefExpr(to(decl().bind("LHS")))),
          hasArgument(1, IgnoringTemporaries(
                             callExpr(callee(StrCat), hasArgument(0, AlphaNum),
                                      unless(HasAnotherReferenceToLhs))
                                 .bind("Call"))))
          .bind("Op"),
      this);
}

void StrCatAppendCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Op = Result.Nodes.getNodeAs<CXXOperatorCallExpr>("Op");
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("Call");
  assert(Op != nullptr && Call != nullptr && "Matcher does not work as expected");

  // Handles the case 'x = absl::StrCat(x)', which has no effect.
  if (Call->getNumArgs() == 1) {
    diag(Op->getBeginLoc(), "call to 'absl::StrCat' has no effect");
    return;
  }

  // Emit a warning and emit fixits to go from
  //   x = absl::StrCat(x, ...)
  // to
  //   absl::StrAppend(&x, ...)
  diag(Op->getBeginLoc(),
       "call 'absl::StrAppend' instead of 'absl::StrCat' when appending to a "
       "string to avoid a performance penalty")
      << FixItHint::CreateReplacement(
             CharSourceRange::getTokenRange(Op->getBeginLoc(),
                                            Call->getCallee()->getEndLoc()),
             "absl::StrAppend")
      << FixItHint::CreateInsertion(Call->getArg(0)->getBeginLoc(), "&");
}

}  // namespace abseil
}  // namespace tidy
}  // namespace clang
