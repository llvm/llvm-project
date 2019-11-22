//===--- NoAutomaticMoveCheck.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoAutomaticMoveCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace performance {

NoAutomaticMoveCheck::NoAutomaticMoveCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowedTypes(
          utils::options::parseStringList(Options.get("AllowedTypes", ""))) {}

void NoAutomaticMoveCheck::registerMatchers(MatchFinder *Finder) {
  // Automatic move exists only for c++11 onwards.
  if (!getLangOpts().CPlusPlus11)
    return;

  const auto ConstLocalVariable =
      varDecl(hasLocalStorage(), unless(hasType(lValueReferenceType())),
              hasType(qualType(
                  isConstQualified(),
                  hasCanonicalType(matchers::isExpensiveToCopy()),
                  unless(hasDeclaration(namedDecl(
                      matchers::matchesAnyListedName(AllowedTypes)))))))
          .bind("vardecl");

  // A matcher for a `DstT::DstT(const Src&)` where DstT also has a
  // `DstT::DstT(Src&&)`.
  const auto LValueRefCtor = cxxConstructorDecl(
      hasParameter(0,
                   hasType(lValueReferenceType(pointee(type().bind("SrcT"))))),
      ofClass(cxxRecordDecl(hasMethod(cxxConstructorDecl(
          hasParameter(0, hasType(rValueReferenceType(
                              pointee(type(equalsBoundNode("SrcT")))))))))));

  Finder->addMatcher(
      returnStmt(
          hasReturnValue(ignoringElidableConstructorCall(ignoringParenImpCasts(
              cxxConstructExpr(hasDeclaration(LValueRefCtor),
                               hasArgument(0, ignoringParenImpCasts(declRefExpr(
                                                  to(ConstLocalVariable)))))
                  .bind("ctor_call"))))),
      this);
}

void NoAutomaticMoveCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Var = Result.Nodes.getNodeAs<VarDecl>("vardecl");
  const auto *CtorCall = Result.Nodes.getNodeAs<Expr>("ctor_call");
  diag(CtorCall->getExprLoc(), "constness of '%0' prevents automatic move")
      << Var->getName();
}

void NoAutomaticMoveCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowedTypes",
                utils::options::serializeStringList(AllowedTypes));
}

} // namespace performance
} // namespace tidy
} // namespace clang
