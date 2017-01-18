//===--- MiscStringCompare.cpp - clang-tidy--------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "StringCompareCheck.h"
#include "../utils/FixItHintUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

static const StringRef CompareMessage = "do not use 'compare' to test equality "
                                        "of strings; use the string equality "
                                        "operator instead";

void StringCompareCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  const auto StrCompare = cxxMemberCallExpr(
      callee(cxxMethodDecl(hasName("compare"),
                           ofClass(classTemplateSpecializationDecl(
                               hasName("::std::basic_string"))))),
      hasArgument(0, expr().bind("str2")), argumentCountIs(1),
      callee(memberExpr().bind("str1")));

  // First and second case: cast str.compare(str) to boolean.
  Finder->addMatcher(implicitCastExpr(hasImplicitDestinationType(booleanType()),
                                      has(StrCompare))
                         .bind("match1"),
                     this);

  // Third and fourth case: str.compare(str) == 0 and str.compare(str) != 0.
  Finder->addMatcher(
      binaryOperator(anyOf(hasOperatorName("=="), hasOperatorName("!=")),
                     hasEitherOperand(StrCompare.bind("compare")),
                     hasEitherOperand(integerLiteral(equals(0)).bind("zero")))
          .bind("match2"),
      this);
}

void StringCompareCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Matched = Result.Nodes.getNodeAs<Stmt>("match1")) {
    diag(Matched->getLocStart(), CompareMessage);
    return;
  }

  if (const auto *Matched = Result.Nodes.getNodeAs<Stmt>("match2")) {
    const ASTContext &Ctx = *Result.Context;

    if (const auto *Zero = Result.Nodes.getNodeAs<Stmt>("zero")) {
      const auto *Str1 = Result.Nodes.getNodeAs<MemberExpr>("str1");
      const auto *Str2 = Result.Nodes.getNodeAs<Stmt>("str2");
      const auto *Compare = Result.Nodes.getNodeAs<Stmt>("compare");

      auto Diag = diag(Matched->getLocStart(), CompareMessage);

      if (Str1->isArrow())
        Diag << FixItHint::CreateInsertion(Str1->getLocStart(), "*");

      Diag << tooling::fixit::createReplacement(*Zero, *Str2, Ctx)
           << tooling::fixit::createReplacement(*Compare, *Str1->getBase(),
                                                Ctx);
    }
  }

  // FIXME: Add fixit to fix the code for case one and two (match1).
}

} // namespace misc
} // namespace tidy
} // namespace clang
