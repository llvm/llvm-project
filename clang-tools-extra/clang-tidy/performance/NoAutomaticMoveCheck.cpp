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

namespace clang::tidy::performance {

namespace {

AST_MATCHER(VarDecl, isNRVOVariable) { return Node.isNRVOVariable(); }

} // namespace

NoAutomaticMoveCheck::NoAutomaticMoveCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowedTypes(
          utils::options::parseStringList(Options.get("AllowedTypes", ""))) {}

void NoAutomaticMoveCheck::registerMatchers(MatchFinder *Finder) {
  const auto NonNrvoConstLocalVariable =
      varDecl(hasLocalStorage(), unless(hasType(lValueReferenceType())),
              unless(isNRVOVariable()),
              hasType(qualType(
                  isConstQualified(),
                  hasCanonicalType(matchers::isExpensiveToCopy()),
                  unless(hasDeclaration(namedDecl(
                      matchers::matchesAnyListedName(AllowedTypes)))))))
          .bind("vardecl");

  // A matcher for a `DstT::DstT(const Src&)` where DstT also has a
  // `DstT::DstT(Src&&)`.
  const auto LValueRefCtor = cxxConstructorDecl(
      hasParameter(0, hasType(hasCanonicalType(
                          lValueReferenceType(pointee(type().bind("SrcT")))))),
      ofClass(cxxRecordDecl(hasMethod(cxxConstructorDecl(
          hasParameter(0, hasType(hasCanonicalType(rValueReferenceType(
                              pointee(type(equalsBoundNode("SrcT"))))))))))));

  // A matcher for `DstT::DstT(const Src&&)`, which typically comes from an
  // instantiation of `template <typename U> DstT::DstT(U&&)`.
  const auto ConstRefRefCtor = cxxConstructorDecl(
      parameterCountIs(1),
      hasParameter(0,
                   hasType(rValueReferenceType(pointee(isConstQualified())))));

  Finder->addMatcher(
      traverse(
          TK_AsIs,
          returnStmt(hasReturnValue(
              ignoringElidableConstructorCall(ignoringParenImpCasts(
                  cxxConstructExpr(
                      hasDeclaration(anyOf(LValueRefCtor, ConstRefRefCtor)),
                      hasArgument(0, ignoringParenImpCasts(declRefExpr(
                                         to(NonNrvoConstLocalVariable)))))
                      .bind("ctor_call")))))),
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

} // namespace clang::tidy::performance
