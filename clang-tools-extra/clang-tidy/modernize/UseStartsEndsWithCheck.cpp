//===--- UseStartsEndsWithCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStartsEndsWithCheck.h"

#include "../utils/ASTUtils.h"
#include "../utils/Matchers.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {
struct NotLengthExprForStringNode {
  NotLengthExprForStringNode(std::string ID, DynTypedNode Node,
                             ASTContext *Context)
      : ID(std::move(ID)), Node(std::move(Node)), Context(Context) {}
  bool operator()(const internal::BoundNodesMap &Nodes) const {
    // Match a string literal and an integer size or strlen() call.
    if (const auto *StringLiteralNode = Nodes.getNodeAs<StringLiteral>(ID)) {
      if (const auto *IntegerLiteralSizeNode = Node.get<IntegerLiteral>()) {
        return StringLiteralNode->getLength() !=
               IntegerLiteralSizeNode->getValue().getZExtValue();
      }

      if (const auto *StrlenNode = Node.get<CallExpr>()) {
        if (StrlenNode->getDirectCallee()->getName() != "strlen" ||
            StrlenNode->getNumArgs() != 1) {
          return true;
        }

        if (const auto *StrlenArgNode = dyn_cast<StringLiteral>(
                StrlenNode->getArg(0)->IgnoreParenImpCasts())) {
          return StrlenArgNode->getLength() != StringLiteralNode->getLength();
        }
      }
    }

    // Match a string variable and a call to length() or size().
    if (const auto *ExprNode = Nodes.getNodeAs<Expr>(ID)) {
      if (const auto *MemberCallNode = Node.get<CXXMemberCallExpr>()) {
        const CXXMethodDecl *MethodDeclNode = MemberCallNode->getMethodDecl();
        const StringRef Name = MethodDeclNode->getName();
        if (!MethodDeclNode->isConst() || MethodDeclNode->getNumParams() != 0 ||
            (Name != "size" && Name != "length")) {
          return true;
        }

        if (const auto *OnNode =
                dyn_cast<Expr>(MemberCallNode->getImplicitObjectArgument())) {
          return !utils::areStatementsIdentical(OnNode->IgnoreParenImpCasts(),
                                                ExprNode->IgnoreParenImpCasts(),
                                                *Context);
        }
      }
    }

    return true;
  }

private:
  std::string ID;
  DynTypedNode Node;
  ASTContext *Context;
};

AST_MATCHER_P(Expr, lengthExprForStringNode, std::string, ID) {
  return Builder->removeBindings(NotLengthExprForStringNode(
      ID, DynTypedNode::create(Node), &(Finder->getASTContext())));
}

UseStartsEndsWithCheck::UseStartsEndsWithCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void UseStartsEndsWithCheck::registerMatchers(MatchFinder *Finder) {
  const auto ZeroLiteral = integerLiteral(equals(0));

  const auto ClassTypeWithMethod = [](const StringRef MethodBoundName,
                                      const auto... Methods) {
    return cxxRecordDecl(anyOf(
        hasMethod(cxxMethodDecl(isConst(), parameterCountIs(1),
                                returns(booleanType()), hasAnyName(Methods))
                      .bind(MethodBoundName))...));
  };

  const auto OnClassWithStartsWithFunction =
      ClassTypeWithMethod("starts_with_fun", "starts_with", "startsWith",
                          "startswith", "StartsWith");

  const auto OnClassWithEndsWithFunction = ClassTypeWithMethod(
      "ends_with_fun", "ends_with", "endsWith", "endswith", "EndsWith");

  // Case 1: X.find(Y) [!=]= 0 -> starts_with.
  const auto FindExpr = cxxMemberCallExpr(
      anyOf(argumentCountIs(1), hasArgument(1, ZeroLiteral)),
      callee(
          cxxMethodDecl(hasName("find"), ofClass(OnClassWithStartsWithFunction))
              .bind("find_fun")),
      hasArgument(0, expr().bind("needle")));

  // Case 2: X.rfind(Y, 0) [!=]= 0 -> starts_with.
  const auto RFindExpr = cxxMemberCallExpr(
      hasArgument(1, ZeroLiteral),
      callee(cxxMethodDecl(hasName("rfind"),
                           ofClass(OnClassWithStartsWithFunction))
                 .bind("find_fun")),
      hasArgument(0, expr().bind("needle")));

  // Case 3: X.compare(0, LEN(Y), Y) [!=]= 0 -> starts_with.
  const auto CompareExpr = cxxMemberCallExpr(
      argumentCountIs(3), hasArgument(0, ZeroLiteral),
      callee(cxxMethodDecl(hasName("compare"),
                           ofClass(OnClassWithStartsWithFunction))
                 .bind("find_fun")),
      hasArgument(2, expr().bind("needle")),
      hasArgument(1, lengthExprForStringNode("needle")));

  // Case 4: X.compare(LEN(X) - LEN(Y), LEN(Y), Y) [!=]= 0 -> ends_with.
  const auto CompareEndsWithExpr = cxxMemberCallExpr(
      argumentCountIs(3),
      callee(cxxMethodDecl(hasName("compare"),
                           ofClass(OnClassWithEndsWithFunction))
                 .bind("find_fun")),
      on(expr().bind("haystack")), hasArgument(2, expr().bind("needle")),
      hasArgument(1, lengthExprForStringNode("needle")),
      hasArgument(0,
                  binaryOperator(hasOperatorName("-"),
                                 hasLHS(lengthExprForStringNode("haystack")),
                                 hasRHS(lengthExprForStringNode("needle")))));

  // All cases comparing to 0.
  Finder->addMatcher(
      binaryOperator(
          matchers::isEqualityOperator(),
          hasOperands(cxxMemberCallExpr(anyOf(FindExpr, RFindExpr, CompareExpr,
                                              CompareEndsWithExpr))
                          .bind("find_expr"),
                      ZeroLiteral))
          .bind("expr"),
      this);

  // Case 5: X.rfind(Y) [!=]= LEN(X) - LEN(Y) -> ends_with.
  Finder->addMatcher(
      binaryOperator(
          matchers::isEqualityOperator(),
          hasOperands(
              cxxMemberCallExpr(
                  anyOf(
                      argumentCountIs(1),
                      allOf(argumentCountIs(2),
                            hasArgument(
                                1,
                                anyOf(declRefExpr(to(varDecl(hasName("npos")))),
                                      memberExpr(member(hasName("npos"))))))),
                  callee(cxxMethodDecl(hasName("rfind"),
                                       ofClass(OnClassWithEndsWithFunction))
                             .bind("find_fun")),
                  on(expr().bind("haystack")),
                  hasArgument(0, expr().bind("needle")))
                  .bind("find_expr"),
              binaryOperator(hasOperatorName("-"),
                             hasLHS(lengthExprForStringNode("haystack")),
                             hasRHS(lengthExprForStringNode("needle")))))
          .bind("expr"),
      this);
}

void UseStartsEndsWithCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ComparisonExpr = Result.Nodes.getNodeAs<BinaryOperator>("expr");
  const auto *FindExpr = Result.Nodes.getNodeAs<CXXMemberCallExpr>("find_expr");
  const auto *FindFun = Result.Nodes.getNodeAs<CXXMethodDecl>("find_fun");
  const auto *SearchExpr = Result.Nodes.getNodeAs<Expr>("needle");
  const auto *StartsWithFunction =
      Result.Nodes.getNodeAs<CXXMethodDecl>("starts_with_fun");
  const auto *EndsWithFunction =
      Result.Nodes.getNodeAs<CXXMethodDecl>("ends_with_fun");
  assert(bool(StartsWithFunction) != bool(EndsWithFunction));
  const CXXMethodDecl *ReplacementFunction =
      StartsWithFunction ? StartsWithFunction : EndsWithFunction;

  if (ComparisonExpr->getBeginLoc().isMacroID())
    return;

  const bool Neg = ComparisonExpr->getOpcode() == BO_NE;

  auto Diagnostic =
      diag(FindExpr->getExprLoc(), "use %0 instead of %1() %select{==|!=}2 0")
      << ReplacementFunction->getName() << FindFun->getName() << Neg;

  // Remove possible arguments after search expression and ' [!=]= .+' suffix.
  Diagnostic << FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(
          Lexer::getLocForEndOfToken(SearchExpr->getEndLoc(), 0,
                                     *Result.SourceManager, getLangOpts()),
          ComparisonExpr->getEndLoc()),
      ")");

  // Remove possible '.+ [!=]= ' prefix.
  Diagnostic << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
      ComparisonExpr->getBeginLoc(), FindExpr->getBeginLoc()));

  // Replace method name by '(starts|ends)_with'.
  // Remove possible arguments before search expression.
  Diagnostic << FixItHint::CreateReplacement(
      CharSourceRange::getCharRange(FindExpr->getExprLoc(),
                                    SearchExpr->getBeginLoc()),
      (ReplacementFunction->getName() + "(").str());

  // Add possible negation '!'.
  if (Neg)
    Diagnostic << FixItHint::CreateInsertion(FindExpr->getBeginLoc(), "!");
}

} // namespace clang::tidy::modernize
