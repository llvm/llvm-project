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

namespace clang {
namespace tidy {
namespace readability {
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

  Finder->addMatcher(
      unaryOperator(
          unless(isExpansionInSystemHeader()), hasOperatorName("&"),
          hasUnaryOperand(anyOf(
              ignoringParenImpCasts(
                  cxxOperatorCallExpr(
                      callee(cxxMethodDecl(hasName("operator[]"))
                                 .bind("operator[]")),
                      argumentCountIs(2),
                      hasArgument(
                          0,
                          anyOf(ignoringParenImpCasts(
                                    declRefExpr(
                                        to(varDecl(anyOf(
                                            hasType(Container),
                                            hasType(references(Container))))))
                                        .bind("var")),
                                ignoringParenImpCasts(hasDescendant(
                                    declRefExpr(
                                        to(varDecl(anyOf(
                                            hasType(Container),
                                            hasType(pointsTo(Container)),
                                            hasType(references(Container))))))
                                        .bind("var"))))),
                      hasArgument(1,
                                  ignoringParenImpCasts(
                                      integerLiteral(equals(0)).bind("zero"))))
                      .bind("operator-call")),
              ignoringParenImpCasts(
                  cxxMemberCallExpr(
                      hasDescendant(
                          declRefExpr(to(varDecl(anyOf(
                                          hasType(Container),
                                          hasType(references(Container))))))
                              .bind("var")),
                      argumentCountIs(1),
                      hasArgument(0,
                                  ignoringParenImpCasts(
                                      integerLiteral(equals(0)).bind("zero"))))
                      .bind("member-call")),
              ignoringParenImpCasts(
                  arraySubscriptExpr(
                      hasLHS(ignoringParenImpCasts(
                          declRefExpr(to(varDecl(anyOf(
                                          hasType(Container),
                                          hasType(references(Container))))))
                              .bind("var"))),
                      hasRHS(ignoringParenImpCasts(
                          integerLiteral(equals(0)).bind("zero"))))
                      .bind("array-subscript")))))
          .bind("address-of"),
      this);
}

void ContainerDataPointerCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *UO = Result.Nodes.getNodeAs<UnaryOperator>("address-of");
  const auto *DRE = Result.Nodes.getNodeAs<DeclRefExpr>("var");

  std::string ReplacementText;
  ReplacementText = std::string(Lexer::getSourceText(
      CharSourceRange::getTokenRange(DRE->getSourceRange()),
      *Result.SourceManager, getLangOpts()));
  if (DRE->getType()->isPointerType())
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
} // namespace readability
} // namespace tidy
} // namespace clang
