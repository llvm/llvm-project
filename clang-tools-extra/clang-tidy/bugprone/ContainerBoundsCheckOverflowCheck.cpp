//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ContainerBoundsCheckOverflowCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

ContainerBoundsCheckOverflowCheck::ContainerBoundsCheckOverflowCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoredContainers(utils::options::parseStringList(
          Options.get("IgnoredContainers", ""))),
      SizeMethodNames(utils::options::parseStringList(
          Options.get("SizeMethodNames", "size;length"))),
      IncludedFreeStandingSizeFuncNames(utils::options::parseStringList(
          Options.get("IncludedFreeStandingSizeFuncNames", "::std::size"))) {}

void ContainerBoundsCheckOverflowCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoredContainers",
                utils::options::serializeStringList(IgnoredContainers));
  Options.store(Opts, "SizeMethodNames",
                utils::options::serializeStringList(SizeMethodNames));
  Options.store(
      Opts, "IncludedFreeStandingSizeFuncNames",
      utils::options::serializeStringList(IncludedFreeStandingSizeFuncNames));
}

void ContainerBoundsCheckOverflowCheck::registerMatchers(MatchFinder *Finder) {
  auto RecordMatcher = cxxRecordDecl();
  if (!IgnoredContainers.empty())
    RecordMatcher = cxxRecordDecl(unless(hasAnyName(IgnoredContainers)));
  auto SizeMethodCall = cxxMemberCallExpr(
      callee(cxxMethodDecl(hasAnyName(SizeMethodNames))),
      on(hasType(hasCanonicalType(hasDeclaration(RecordMatcher)))));
  auto FreeStandingSizeCall = callExpr(
      callee(functionDecl(hasAnyName(IncludedFreeStandingSizeFuncNames))));
  // The size can either be a direct member size method call, a free-standing
  // size function call, or a reference to a variable initialized from one (e.g.
  // auto s = v.size();)
  auto SizeExpr = ignoringParenImpCasts(
      expr(anyOf(SizeMethodCall, FreeStandingSizeCall,
                 declRefExpr(to(varDecl(
                     hasInitializer(ignoringParenImpCasts(SizeMethodCall)))))))
          .bind("size_expr"));

  // Operands must be unsigned integers, as overflow in signed integer addition
  // is undefined behavior
  auto Addition =
      binaryOperator(hasOperatorName("+"), hasLHS(hasType(isUnsignedInteger())),
                     hasRHS(hasType(isUnsignedInteger())))
          .bind("addition");
  auto Comparison = hasAnyOperatorName("<", "<=", ">", ">=");
  // Match cases: [Addition] </<=/>/>= [Size]
  Finder->addMatcher(binaryOperator(Comparison, hasLHS(Addition),
                                    hasRHS(ignoringParenImpCasts(SizeExpr)))
                         .bind("comparison_addition_lhs"),
                     this);
  // Match cases: [Size] </<=/>/>= [Addition]
  Finder->addMatcher(binaryOperator(Comparison, hasRHS(Addition),
                                    hasLHS(ignoringParenImpCasts(SizeExpr)))
                         .bind("comparison_addition_rhs"),
                     this);
}

void ContainerBoundsCheckOverflowCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Addition = Result.Nodes.getNodeAs<BinaryOperator>("addition");
  const auto *SizeExpr = Result.Nodes.getNodeAs<Expr>("size_expr");
  if (!Addition || !SizeExpr)
    return;
  const auto *ComparisonAddLhs =
      Result.Nodes.getNodeAs<BinaryOperator>("comparison_addition_lhs");
  const auto *ComparisonAddRhs =
      Result.Nodes.getNodeAs<BinaryOperator>("comparison_addition_rhs");
  const auto NoComparison = !ComparisonAddLhs && !ComparisonAddRhs;
  if (NoComparison)
    return;

  auto AdditionType = Addition->getType().getCanonicalType();
  auto SizeExprType = SizeExpr->getType().getCanonicalType();

  auto &Context = *Result.Context;
  // If the type of the addition is smaller than the type of the size() call,
  // then the addition will be promoted to the size() type before the
  // comparison, so there is no risk of overflow. The case where the type of the
  // addition is larger than the type of the size() call is not handled by this
  // check
  if (Context.getTypeSize(AdditionType) != Context.getTypeSize(SizeExprType))
    return;

  const auto *Comparison =
      ComparisonAddLhs ? ComparisonAddLhs : ComparisonAddRhs;

  auto Diag = diag(Comparison->getOperatorLoc(),
                   "potential overflow in unsigned integer addition "
                   "before comparison");

  // A fix is only produced when both operands of the addition are simple
  // expressions. If an operand is itself a compound expression (e.g. 'a + b' in
  // 'a + b + c'), only a diagnostic is emitted, as rewriting such cases is
  // error-prone
  const bool HasCompoundOperand =
      isa<BinaryOperator>(Addition->getLHS()->IgnoreParenImpCasts()) ||
      isa<BinaryOperator>(Addition->getRHS()->IgnoreParenImpCasts());
  if (HasCompoundOperand)
    return;

  // Introduce parentheses around the addition to avoid changing the order of
  // operations when replacing the comparison with a logical AND/OR expression.
  // The parentheses are only added if the original expression is not already
  // wrapped in parentheses
  bool NeedsParens = true;
  const auto &Parents = Context.getParents(*Comparison);
  if (!Parents.empty()) {
    if (Parents[0].get<ParenExpr>() || Parents[0].get<IfStmt>() ||
        Parents[0].get<WhileStmt>())
      NeedsParens = false;
  }

  auto GetText = [&](SourceRange Range) -> StringRef {
    return Lexer::getSourceText(CharSourceRange::getTokenRange(Range),
                                *Result.SourceManager, getLangOpts());
  };
  const std::string StrA = GetText(Addition->getLHS()->getSourceRange()).str();
  const std::string StrB = GetText(Addition->getRHS()->getSourceRange()).str();
  const std::string StrSize = GetText(SizeExpr->getSourceRange()).str();

  const auto ComparisonType = Comparison->getOpcodeStr();
  std::string Replacement;
  if (ComparisonAddLhs) {
    // Matches cases where the addition is on the left side of the comparison
    // (a + b < size())  -> (a < size() && b < size() - a)
    // (a + b <= size()) -> (a <= size() && b <= size() - a)
    // (a + b > size())  -> (a > size() || b > size() - a)
    // (a + b >= size()) -> (a >= size() || b >= size() - a)
    const auto *Expr =
        (ComparisonType == "<" || ComparisonType == "<=") ? " && " : " || ";
    Replacement = (NeedsParens ? "(" : "") + StrA + " " + ComparisonType.str() +
                  " " + StrSize + Expr + StrB + " " + ComparisonType.str() +
                  " " + StrSize + " - " + StrA + (NeedsParens ? ")" : "");
  } else {
    // Matches cases where the addition is on the right side of the comparison,
    // (size() < a + b)  -> (size() < a || size() - a < b)
    // (size() <= a + b) -> (size() <= a || size() - a <= b)
    // (size() > a + b)  -> (size() > a && size() - a > b)
    // (size() >= a + b) -> (size() >= a && size() - a >= b)
    const auto *Expr =
        (ComparisonType == "<" || ComparisonType == "<=") ? " || " : " && ";
    Replacement = (NeedsParens ? "(" : "") + StrSize + " " +
                  ComparisonType.str() + " " + StrA + Expr + StrSize + " - " +
                  StrA + " " + ComparisonType.str() + " " + StrB +
                  (NeedsParens ? ")" : "");
  }

  Diag << FixItHint::CreateReplacement(Comparison->getSourceRange(),
                                       Replacement);
}

} // namespace clang::tidy::bugprone
