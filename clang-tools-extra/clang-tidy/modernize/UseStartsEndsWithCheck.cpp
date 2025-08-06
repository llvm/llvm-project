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

static bool isNegativeComparison(const Expr *ComparisonExpr) {
  if (const auto *Op = llvm::dyn_cast<BinaryOperator>(ComparisonExpr))
    return Op->getOpcode() == BO_NE;

  if (const auto *Op = llvm::dyn_cast<CXXOperatorCallExpr>(ComparisonExpr))
    return Op->getOperator() == OO_ExclaimEqual;

  if (const auto *Op =
          llvm::dyn_cast<CXXRewrittenBinaryOperator>(ComparisonExpr))
    return Op->getOperator() == BO_NE;

  return false;
}

namespace {

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

} // namespace

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

  // Case 1: X.find(Y, [0], [LEN(Y)]) [!=]= 0 -> starts_with.
  const auto FindExpr = cxxMemberCallExpr(
      callee(
          cxxMethodDecl(hasName("find"), ofClass(OnClassWithStartsWithFunction))
              .bind("find_fun")),
      hasArgument(0, expr().bind("needle")),
      anyOf(
          // Detect the expression: X.find(Y);
          argumentCountIs(1),
          // Detect the expression: X.find(Y, 0);
          allOf(argumentCountIs(2), hasArgument(1, ZeroLiteral)),
          // Detect the expression: X.find(Y, 0, LEN(Y));
          allOf(argumentCountIs(3), hasArgument(1, ZeroLiteral),
                hasArgument(2, lengthExprForStringNode("needle")))));

  // Case 2: X.rfind(Y, 0, [LEN(Y)]) [!=]= 0 -> starts_with.
  const auto RFindExpr = cxxMemberCallExpr(
      callee(cxxMethodDecl(hasName("rfind"),
                           ofClass(OnClassWithStartsWithFunction))
                 .bind("find_fun")),
      hasArgument(0, expr().bind("needle")),
      anyOf(
          // Detect the expression: X.rfind(Y, 0);
          allOf(argumentCountIs(2), hasArgument(1, ZeroLiteral)),
          // Detect the expression: X.rfind(Y, 0, LEN(Y));
          allOf(argumentCountIs(3), hasArgument(1, ZeroLiteral),
                hasArgument(2, lengthExprForStringNode("needle")))));

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

  // Case 6: X.substr(0, LEN(Y)) [!=]= Y -> starts_with.
  Finder->addMatcher(
      binaryOperation(
          hasAnyOperatorName("==", "!="),
          hasOperands(
              expr().bind("needle"),
              cxxMemberCallExpr(
                  argumentCountIs(2), hasArgument(0, ZeroLiteral),
                  hasArgument(1, lengthExprForStringNode("needle")),
                  callee(cxxMethodDecl(hasName("substr"),
                                       ofClass(OnClassWithStartsWithFunction))
                             .bind("find_fun")))
                  .bind("find_expr")))
          .bind("expr"),
      this);
}

void UseStartsEndsWithCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ComparisonExpr = Result.Nodes.getNodeAs<Expr>("expr");
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

  if (ComparisonExpr->getBeginLoc().isMacroID() ||
      FindExpr->getBeginLoc().isMacroID())
    return;

  // Make sure FindExpr->getArg(0) can be used to make a range in the FitItHint.
  if (FindExpr->getNumArgs() == 0)
    return;

  // Retrieve the source text of the search expression.
  const auto SearchExprText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(SearchExpr->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts());

  auto Diagnostic = diag(FindExpr->getExprLoc(), "use %0 instead of %1")
                    << ReplacementFunction->getName() << FindFun->getName();

  // Remove everything before the function call.
  Diagnostic << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
      ComparisonExpr->getBeginLoc(), FindExpr->getBeginLoc()));

  // Rename the function to `starts_with` or `ends_with`.
  Diagnostic << FixItHint::CreateReplacement(FindExpr->getExprLoc(),
                                             ReplacementFunction->getName());

  // Replace arguments and everything after the function call.
  Diagnostic << FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(FindExpr->getArg(0)->getBeginLoc(),
                                     ComparisonExpr->getEndLoc()),
      (SearchExprText + ")").str());

  // Add negation if necessary.
  if (isNegativeComparison(ComparisonExpr))
    Diagnostic << FixItHint::CreateInsertion(FindExpr->getBeginLoc(), "!");
}

} // namespace clang::tidy::modernize
