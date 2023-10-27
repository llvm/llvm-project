//===--- MissingStdForwardCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MissingStdForwardCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

namespace {

using matchers::hasUnevaluatedContext;

AST_MATCHER_P(QualType, possiblyPackExpansionOf,
              ast_matchers::internal::Matcher<QualType>, InnerMatcher) {
  return InnerMatcher.matches(Node.getNonPackExpansionType(), Finder, Builder);
}

AST_MATCHER(ParmVarDecl, isTemplateTypeParameter) {
  ast_matchers::internal::Matcher<QualType> Inner = possiblyPackExpansionOf(
      qualType(rValueReferenceType(),
               references(templateTypeParmType(
                   hasDeclaration(templateTypeParmDecl()))),
               unless(references(qualType(isConstQualified())))));
  if (!Inner.matches(Node.getType(), Finder, Builder))
    return false;

  const auto *Function = dyn_cast<FunctionDecl>(Node.getDeclContext());
  if (!Function)
    return false;

  const FunctionTemplateDecl *FuncTemplate =
      Function->getDescribedFunctionTemplate();
  if (!FuncTemplate)
    return false;

  QualType ParamType =
      Node.getType().getNonPackExpansionType()->getPointeeType();
  const auto *TemplateType = ParamType->getAs<TemplateTypeParmType>();
  if (!TemplateType)
    return false;

  return TemplateType->getDepth() ==
         FuncTemplate->getTemplateParameters()->getDepth();
}

} // namespace

void MissingStdForwardCheck::registerMatchers(MatchFinder *Finder) {
  auto ToParam = hasAnyParameter(parmVarDecl(equalsBoundNode("param")));

  auto ForwardCallMatcher = callExpr(
      forCallable(equalsBoundNode("func")), argumentCountIs(1),
      callee(unresolvedLookupExpr(hasAnyDeclaration(
          namedDecl(hasUnderlyingDecl(hasName("::std::forward")))))),
      hasArgument(0, declRefExpr(to(equalsBoundNode("param"))).bind("ref")),
      unless(anyOf(hasAncestor(typeLoc()),
                   hasAncestor(expr(hasUnevaluatedContext())))));

  Finder->addMatcher(
      parmVarDecl(parmVarDecl().bind("param"), isTemplateTypeParameter(),
                  hasAncestor(functionDecl().bind("func")),
                  hasAncestor(functionDecl(
                      isDefinition(), equalsBoundNode("func"), ToParam,
                      unless(hasDescendant(std::move(ForwardCallMatcher)))))),
      this);
}

void MissingStdForwardCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");

  if (!Param)
    return;

  diag(Param->getLocation(),
       "forwarding reference parameter %0 is never forwarded "
       "inside the function body")
      << Param;
}

} // namespace clang::tidy::cppcoreguidelines
