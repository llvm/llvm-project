//===--- ContainerDataPointerCheck.cpp - clang-tidy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ContainerDataPointerCheck.h"

#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringRef.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

constexpr llvm::StringLiteral ContainerExprName = "container-expr";
constexpr llvm::StringLiteral DerefContainerExprName = "deref-container-expr";
constexpr llvm::StringLiteral AddrOfContainerExprName =
    "addr-of-container-expr";
constexpr llvm::StringLiteral AddressOfName = "address-of";

ContainerDataPointerCheck::ContainerDataPointerCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void ContainerDataPointerCheck::registerMatchers(MatchFinder *Finder) {
  const auto Record =
      cxxRecordDecl(
          isSameOrDerivedFrom(
              namedDecl(
                  has(cxxMethodDecl(isPublic(), hasName("data")).bind("data")))
                  .bind("container")))
          .bind("record");

  const auto NonTemplateContainerType =
      qualType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(Record))));
  const auto TemplateContainerType =
      qualType(hasUnqualifiedDesugaredType(templateSpecializationType(
          hasDeclaration(classTemplateDecl(has(Record))))));

  const auto Container =
      qualType(anyOf(NonTemplateContainerType, TemplateContainerType));

  const auto ContainerExpr = anyOf(
      unaryOperator(
          hasOperatorName("*"),
          hasUnaryOperand(
              expr(hasType(pointsTo(Container))).bind(DerefContainerExprName)))
          .bind(ContainerExprName),
      unaryOperator(hasOperatorName("&"),
                    hasUnaryOperand(expr(anyOf(hasType(Container),
                                               hasType(references(Container))))
                                        .bind(AddrOfContainerExprName)))
          .bind(ContainerExprName),
      expr(anyOf(hasType(Container), hasType(pointsTo(Container)),
                 hasType(references(Container))))
          .bind(ContainerExprName));

  const auto Zero = integerLiteral(equals(0));

  const auto SubscriptOperator = callee(cxxMethodDecl(hasName("operator[]")));

  Finder->addMatcher(
      unaryOperator(
          unless(isExpansionInSystemHeader()), hasOperatorName("&"),
          hasUnaryOperand(expr(
              anyOf(cxxOperatorCallExpr(SubscriptOperator, argumentCountIs(2),
                                        hasArgument(0, ContainerExpr),
                                        hasArgument(1, Zero)),
                    cxxMemberCallExpr(SubscriptOperator, on(ContainerExpr),
                                      argumentCountIs(1), hasArgument(0, Zero)),
                    arraySubscriptExpr(hasLHS(ContainerExpr), hasRHS(Zero))))))
          .bind(AddressOfName),
      this);
}

void ContainerDataPointerCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *UO = Result.Nodes.getNodeAs<UnaryOperator>(AddressOfName);
  const auto *CE = Result.Nodes.getNodeAs<Expr>(ContainerExprName);
  const auto *DCE = Result.Nodes.getNodeAs<Expr>(DerefContainerExprName);
  const auto *ACE = Result.Nodes.getNodeAs<Expr>(AddrOfContainerExprName);

  if (!UO || !CE)
    return;

  if (DCE && !CE->getType()->isPointerType())
    CE = DCE;
  else if (ACE)
    CE = ACE;

  SourceRange SrcRange = CE->getSourceRange();

  std::string ReplacementText{
      Lexer::getSourceText(CharSourceRange::getTokenRange(SrcRange),
                           *Result.SourceManager, getLangOpts())};

  if (!isa<DeclRefExpr, ArraySubscriptExpr, CXXOperatorCallExpr, CallExpr,
           MemberExpr>(CE))
    ReplacementText = "(" + ReplacementText + ")";

  if (CE->getType()->isPointerType())
    ReplacementText += "->data()";
  else
    ReplacementText += ".data()";

  FixItHint Hint =
      FixItHint::CreateReplacement(UO->getSourceRange(), ReplacementText);
  diag(UO->getBeginLoc(),
       "'data' should be used for accessing the data pointer instead of taking "
       "the address of the 0-th element")
      << Hint;
}
} // namespace clang::tidy::readability
