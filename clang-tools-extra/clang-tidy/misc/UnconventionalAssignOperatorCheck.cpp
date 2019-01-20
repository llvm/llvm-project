//===--- UnconventionalAssignOperatorCheck.cpp - clang-tidy -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnconventionalAssignOperatorCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

void UnconventionalAssignOperatorCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  // Only register the matchers for C++; the functionality currently does not
  // provide any benefit to other languages, despite being benign.
  if (!getLangOpts().CPlusPlus)
    return;

  const auto HasGoodReturnType = cxxMethodDecl(returns(lValueReferenceType(
      pointee(unless(isConstQualified()),
              anyOf(autoType(), hasDeclaration(equalsBoundNode("class")))))));

  const auto IsSelf = qualType(
      anyOf(hasDeclaration(equalsBoundNode("class")),
            referenceType(pointee(hasDeclaration(equalsBoundNode("class"))))));
  const auto IsAssign =
      cxxMethodDecl(unless(anyOf(isDeleted(), isPrivate(), isImplicit())),
                    hasName("operator="), ofClass(recordDecl().bind("class")))
          .bind("method");
  const auto IsSelfAssign =
      cxxMethodDecl(IsAssign, hasParameter(0, parmVarDecl(hasType(IsSelf))))
          .bind("method");

  Finder->addMatcher(
      cxxMethodDecl(IsAssign, unless(HasGoodReturnType)).bind("ReturnType"),
      this);

  const auto BadSelf = referenceType(
      anyOf(lValueReferenceType(pointee(unless(isConstQualified()))),
            rValueReferenceType(pointee(isConstQualified()))));

  Finder->addMatcher(
      cxxMethodDecl(IsSelfAssign,
                    hasParameter(0, parmVarDecl(hasType(BadSelf))))
          .bind("ArgumentType"),
      this);

  Finder->addMatcher(
      cxxMethodDecl(IsSelfAssign, anyOf(isConst(), isVirtual())).bind("cv"),
      this);

  const auto IsBadReturnStatement = returnStmt(unless(has(ignoringParenImpCasts(
      anyOf(unaryOperator(hasOperatorName("*"), hasUnaryOperand(cxxThisExpr())),
            cxxOperatorCallExpr(argumentCountIs(1),
                                callee(unresolvedLookupExpr()),
                                hasArgument(0, cxxThisExpr())))))));
  const auto IsGoodAssign = cxxMethodDecl(IsAssign, HasGoodReturnType);

  Finder->addMatcher(returnStmt(IsBadReturnStatement, forFunction(IsGoodAssign))
                         .bind("returnStmt"),
                     this);
}

void UnconventionalAssignOperatorCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *RetStmt = Result.Nodes.getNodeAs<ReturnStmt>("returnStmt")) {
    diag(RetStmt->getBeginLoc(), "operator=() should always return '*this'");
  } else {
    static const char *const Messages[][2] = {
        {"ReturnType", "operator=() should return '%0&'"},
        {"ArgumentType", "operator=() should take '%0 const&', '%0&&' or '%0'"},
        {"cv", "operator=() should not be marked '%1'"}};

    const auto *Method = Result.Nodes.getNodeAs<CXXMethodDecl>("method");
    for (const auto &Message : Messages) {
      if (Result.Nodes.getNodeAs<Decl>(Message[0]))
        diag(Method->getBeginLoc(), Message[1])
            << Method->getParent()->getName()
            << (Method->isConst() ? "const" : "virtual");
    }
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang
