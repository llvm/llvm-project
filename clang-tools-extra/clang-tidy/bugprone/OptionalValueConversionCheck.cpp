//===--- OptionalValueConversionCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OptionalValueConversionCheck.h"
#include "../utils/LexerUtils.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <array>

using namespace clang::ast_matchers;
using clang::ast_matchers::internal::Matcher;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER_P(QualType, hasCleanType, Matcher<QualType>, InnerMatcher) {
  return InnerMatcher.matches(
      Node.getNonReferenceType().getUnqualifiedType().getCanonicalType(),
      Finder, Builder);
}

constexpr std::array<StringRef, 2> NameList{
    "::std::make_unique",
    "::std::make_shared",
};

Matcher<Expr> constructFrom(Matcher<QualType> TypeMatcher,
                            Matcher<Expr> ArgumentMatcher) {
  return expr(
      anyOf(
          // construct optional
          cxxConstructExpr(argumentCountIs(1U), hasType(TypeMatcher),
                           hasArgument(0U, ArgumentMatcher)),
          // known template methods in std
          callExpr(argumentCountIs(1),
                   callee(functionDecl(
                       matchers::matchesAnyListedName(NameList),
                       hasTemplateArgument(0, refersToType(TypeMatcher)))),
                   hasArgument(0, ArgumentMatcher))),
      unless(anyOf(hasAncestor(typeLoc()),
                   hasAncestor(expr(matchers::hasUnevaluatedContext())))));
}

} // namespace

OptionalValueConversionCheck::OptionalValueConversionCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      OptionalTypes(utils::options::parseStringList(
          Options.get("OptionalTypes",
                      "::std::optional;::absl::optional;::boost::optional"))),
      ValueMethods(utils::options::parseStringList(
          Options.get("ValueMethods", "::value$;::get$"))) {}

std::optional<TraversalKind>
OptionalValueConversionCheck::getCheckTraversalKind() const {
  return TK_AsIs;
}

void OptionalValueConversionCheck::registerMatchers(MatchFinder *Finder) {
  auto BindOptionalType = qualType(
      hasCleanType(qualType(hasDeclaration(namedDecl(
                                matchers::matchesAnyListedName(OptionalTypes))))
                       .bind("optional-type")));

  auto EqualsBoundOptionalType =
      qualType(hasCleanType(equalsBoundNode("optional-type")));

  auto OptionalDereferenceMatcher = callExpr(
      anyOf(
          cxxOperatorCallExpr(hasOverloadedOperatorName("*"),
                              hasUnaryOperand(hasType(EqualsBoundOptionalType)))
              .bind("op-call"),
          cxxMemberCallExpr(thisPointerType(EqualsBoundOptionalType),
                            callee(cxxMethodDecl(anyOf(
                                hasOverloadedOperatorName("*"),
                                matchers::matchesAnyListedName(ValueMethods)))))
              .bind("member-call")),
      hasType(qualType().bind("value-type")));

  auto StdMoveCallMatcher =
      callExpr(argumentCountIs(1), callee(functionDecl(hasName("::std::move"))),
               hasArgument(0, ignoringImpCasts(OptionalDereferenceMatcher)));
  Finder->addMatcher(
      expr(constructFrom(BindOptionalType,
                         ignoringImpCasts(anyOf(OptionalDereferenceMatcher,
                                                StdMoveCallMatcher))))
          .bind("expr"),
      this);
}

void OptionalValueConversionCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "OptionalTypes",
                utils::options::serializeStringList(OptionalTypes));
  Options.store(Opts, "ValueMethods",
                utils::options::serializeStringList(ValueMethods));
}

void OptionalValueConversionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<Expr>("expr");
  const auto *OptionalType = Result.Nodes.getNodeAs<QualType>("optional-type");
  const auto *ValueType = Result.Nodes.getNodeAs<QualType>("value-type");

  diag(MatchedExpr->getExprLoc(),
       "conversion from %0 into %1 and back into %0, remove potentially "
       "error-prone optional dereference")
      << *OptionalType << ValueType->getUnqualifiedType();

  if (const auto *OperatorExpr =
          Result.Nodes.getNodeAs<CXXOperatorCallExpr>("op-call")) {
    diag(OperatorExpr->getExprLoc(), "remove '*' to silence this warning",
         DiagnosticIDs::Note)
        << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
               OperatorExpr->getBeginLoc(), OperatorExpr->getExprLoc()));
    return;
  }
  if (const auto *CallExpr =
          Result.Nodes.getNodeAs<CXXMemberCallExpr>("member-call")) {
    const SourceLocation Begin =
        utils::lexer::getPreviousToken(CallExpr->getExprLoc(),
                                       *Result.SourceManager, getLangOpts())
            .getLocation();
    auto Diag =
        diag(CallExpr->getExprLoc(),
             "remove call to %0 to silence this warning", DiagnosticIDs::Note);
    Diag << CallExpr->getMethodDecl()
         << FixItHint::CreateRemoval(
                CharSourceRange::getTokenRange(Begin, CallExpr->getEndLoc()));
    if (const auto *Member =
            llvm::dyn_cast<MemberExpr>(CallExpr->getCallee()->IgnoreImplicit());
        Member && Member->isArrow())
      Diag << FixItHint::CreateInsertion(CallExpr->getBeginLoc(), "*");
    return;
  }
}

} // namespace clang::tidy::bugprone
