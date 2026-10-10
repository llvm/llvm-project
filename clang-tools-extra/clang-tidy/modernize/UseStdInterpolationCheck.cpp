//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStdInterpolationCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"

using namespace clang::ast_matchers;
using namespace clang::transformer;

namespace clang::tidy::modernize {
namespace {

AST_MATCHER(Expr, hasSideEffects) {
  return Node.HasSideEffects(Finder->getASTContext());
}

AST_MATCHER_P(FloatingLiteral, hasExactValue, double, Value) {
  return Node.getValue().isExactlyValue(Value);
}

AST_MATCHER(Expr, isMacroExpanded) {
  if (Node.getBeginLoc().isMacroID() || Node.getEndLoc().isMacroID())
    return true;
  if (const auto *Operator = dyn_cast<BinaryOperator>(&Node))
    return Operator->getOperatorLoc().isMacroID();
  return false;
}

AST_MATCHER_P(QualType, hasUnqualifiedCanonicalType,
              ast_matchers::internal::Matcher<QualType>, InnerMatcher) {
  return InnerMatcher.matches(Node.getCanonicalType().getUnqualifiedType(),
                              Finder, Builder);
}

AST_MATCHER(QualType, isInterpolationFloatingType) {
  return Node->isSpecificBuiltinType(BuiltinType::Float) ||
         Node->isSpecificBuiltinType(BuiltinType::Double) ||
         Node->isSpecificBuiltinType(BuiltinType::LongDouble);
}

AST_MATCHER(QualType, isInterpolationIntegerType) {
  return Node->isSpecificBuiltinType(BuiltinType::Int) ||
         Node->isSpecificBuiltinType(BuiltinType::UInt) ||
         Node->isSpecificBuiltinType(BuiltinType::Long) ||
         Node->isSpecificBuiltinType(BuiltinType::ULong) ||
         Node->isSpecificBuiltinType(BuiltinType::LongLong) ||
         Node->isSpecificBuiltinType(BuiltinType::ULongLong);
}

using BinaryOperatorMatcher = ast_matchers::internal::Matcher<BinaryOperator>;

struct InterpolationMatchers {
  BinaryOperatorMatcher Midpoint;
  BinaryOperatorMatcher Lerp;
};

} // namespace

static InterpolationMatchers makeInterpolationMatchers() {
  const auto SameType = hasType(qualType(hasUnqualifiedCanonicalType(
      qualType(equalsBoundNode("calculationType")))));
  const auto Start = ignoringParenImpCasts(expr(SameType).bind("start"));
  const auto End = ignoringParenImpCasts(expr(SameType).bind("end"));
  const auto Factor = ignoringParenImpCasts(expr(SameType).bind("factor"));
  const auto RepeatedStart = ignoringParenImpCasts(
      expr(matchers::isStatementIdenticalToBoundNode("start")));
  const auto RepeatedFactor = ignoringParenImpCasts(
      expr(matchers::isStatementIdenticalToBoundNode("factor")));
  const auto Two = ignoringParenImpCasts(
      expr(anyOf(integerLiteral(equals(2)), floatLiteral(hasExactValue(2.0)))));
  const auto Half = ignoringParenImpCasts(floatLiteral(hasExactValue(0.5)));
  const auto One = ignoringParenImpCasts(
      expr(anyOf(integerLiteral(equals(1)), floatLiteral(hasExactValue(1.0)))));

  // Midpoints: (a + b) / 2 and a + (b - a) / 2, also using * 0.5.
  // Exclude ungrouped additive chains, which may express rounded averages
  // rather than an intended two-endpoint midpoint.
  const auto GroupedEndpoint =
      unless(ignoringImpCasts(binaryOperator(hasAnyOperatorName("+", "-"))));
  const auto Sum = ignoringParenImpCasts(binaryOperator(
      hasOperatorName("+"), SameType, hasLHS(expr(GroupedEndpoint, Start)),
      hasRHS(expr(GroupedEndpoint, End))));
  const auto Difference = ignoringParenImpCasts(binaryOperator(
      hasOperatorName("-"), SameType, hasLHS(End), hasRHS(RepeatedStart)));
  const auto HalfSum = binaryOperator(
      anyOf(allOf(hasOperatorName("/"), hasLHS(Sum), hasRHS(Two)),
            allOf(hasOperatorName("*"), hasOperands(Sum, Half))));
  const auto HalfDifference = ignoringParenImpCasts(binaryOperator(
      anyOf(allOf(hasOperatorName("/"), hasLHS(Difference), hasRHS(Two)),
            allOf(hasOperatorName("*"), hasOperands(Difference, Half)))));
  const auto DifferenceMidpoint =
      binaryOperator(hasOperatorName("+"), hasOperands(Start, HalfDifference));

  // Difference-form interpolation: a + (b - a) * t.
  const auto ScaledDifference = ignoringParenImpCasts(binaryOperator(
      hasOperatorName("*"), SameType, hasOperands(Difference, Factor)));
  const auto DifferenceLerp = binaryOperator(
      hasOperatorName("+"), hasOperands(Start, ScaledDifference));

  // Weighted interpolation: (1 - t) * a + t * b.
  const auto Complement = ignoringParenImpCasts(binaryOperator(
      hasOperatorName("-"), SameType, hasLHS(One), hasRHS(Factor)));
  const auto WeightedStart = ignoringParenImpCasts(binaryOperator(
      hasOperatorName("*"), SameType, hasOperands(Start, Complement)));
  const auto WeightedEnd = ignoringParenImpCasts(binaryOperator(
      hasOperatorName("*"), SameType, hasOperands(End, RepeatedFactor)));
  const auto WeightedLerp = binaryOperator(
      hasOperatorName("+"), hasOperands(WeightedStart, WeightedEnd));

  const auto NumericType = hasType(qualType(
      hasUnqualifiedCanonicalType(qualType(anyOf(isInterpolationIntegerType(),
                                                 isInterpolationFloatingType()))
                                      .bind("calculationType"))));
  const auto FloatingType = hasType(qualType(hasUnqualifiedCanonicalType(
      qualType(isInterpolationFloatingType()).bind("calculationType"))));
  const auto Midpoint =
      binaryOperator(NumericType, anyOf(HalfSum, DifferenceMidpoint));
  const auto Lerp =
      binaryOperator(FloatingType, anyOf(DifferenceLerp, WeightedLerp));
  return {Midpoint, Lerp};
}

static BinaryOperatorMatcher eligibleCalculation() {
  return binaryOperator(
      unless(isExpansionInSystemHeader()), unless(isInTemplateInstantiation()),
      unless(isTypeDependent()), unless(isValueDependent()),
      unless(isMacroExpanded()), unless(hasSideEffects()),
      unless(hasDescendant(expr(isMacroExpanded()))),
      unless(hasDescendant(cxxOperatorCallExpr())),
      unless(hasDescendant(cxxMemberCallExpr(callee(cxxConversionDecl())))),
      unless(hasAncestor(expr(matchers::hasUnevaluatedContext()))),
      unless(hasAncestor(typeLoc())));
}

static RewriteRuleWith<std::string> makeInterpolationRule() {
  const auto Patterns = makeInterpolationMatchers();
  const auto Eligible = eligibleCalculation();
  const auto Pattern = binaryOperator(anyOf(Patterns.Midpoint, Patterns.Lerp));
  // Only replace innermost calculations to avoid overlapping edits. The
  // bindings used to test descendants must not escape into the replacement.
  const auto Innermost =
      unless(hasDescendant(binaryOperator(Pattern, Eligible)));

  return applyFirst(
      {makeRule(
           binaryOperator(Patterns.Midpoint, Eligible, Innermost),
           {changeTo(cat("std::midpoint(", expression("start"), ", ",
                         expression("end"), ")")),
            addInclude("numeric", IncludeFormat::Angled)},
           cat("use 'std::midpoint' instead of manual midpoint calculation")),
       makeRule(
           binaryOperator(Patterns.Lerp, Eligible, Innermost),
           {changeTo(cat("std::lerp(", expression("start"), ", ",
                         expression("end"), ", ", expression("factor"), ")")),
            addInclude("cmath", IncludeFormat::Angled)},
           cat("use 'std::lerp' instead of manual linear interpolation"))});
}

UseStdInterpolationCheck::UseStdInterpolationCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : utils::TransformerClangTidyCheck(makeInterpolationRule(), Name, Context) {
}

} // namespace clang::tidy::modernize
