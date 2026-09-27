//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseBulkInsertCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

void UseBulkInsertCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxForRangeStmt(
          hasLoopVariable(varDecl().bind("loop_var")),
          hasRangeInit(expr().bind("range")),
          hasBody(compoundStmt(
              statementCountIs(1),
              hasAnySubstatement(
                cxxMemberCallExpr(
                    callee(memberExpr(ofClass(cxxRecordDecl(hasAnyName(
                        "::std::set", "::std::map", "::std::multiset",
                        "::std::multimap", "::std::unordered_set",
                        "::std::unordered_map", "::std::unordered_multiset",
                        "::std::unordered_multimap"))))),
                    argumentCountIs(1),
                    hasArgument(0, ignoringParenImpCasts(declRefExpr(
                                    to(varDecl().bind("insert_arg"))))))
                        .bind("insert_call")))
          .bind("for_range"),
      this);
}

void UseBulkInsertCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Loop = Result.Nodes.getNodeAs<CXXForRangeStmt>("for_range");
  const auto *LoopVar = Result.Nodes.getNodeAs<VarDecl>("loop_var");
  const auto *InsertArg = Result.Nodes.getNodeAs<VarDecl>("insert_arg");
  const auto *Range = Result.Nodes.getNodeAs<Expr>("range");
  const auto *InsertCall = Result.Nodes.getNodeAs<CXXMemberCallExpr>("insert_call");

  if (!Loop || !LoopVar || !InsertArg || !Range || !InsertCall)
    return;

  if (LoopVar != InsertArg)
    return;

  const auto *Member = dyn_cast<MemberExpr>(InsertCall->getCallee());
  if (!Member)
    return;

  const Expr *Object = Member->getBase();
  if (!Object)
    return;

  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = Result.Context->getLangOpts();

  StringRef ObjectText = Lexer::getSourceText(
    CharSourceRange::getTokenRange(Object->getSourceRange()), SM, LangOpts);

  StringRef RangeText = Lexer::getSourceText(
    CharSourceRange::getTokenRange(Loop->getBeginLoc(),
                              Loop->getEndLoc()),

  if (ObjectText.empty() || RangeText.empty())
    return;

  std::string Replacement = ObjectText.str();
  Replacement += ".insert(";
  Replacement += RangeText;
  Replacement += ".begin(), ";
  Replacement += RangeText;
  Replacement += ".end());";

  diag(Loop->getForLoc(),
       "use bulk insertion instead of inserting elements one at a time")
      << FixItHint::CreateReplacement(
             CharSourceRange::getTokenRange(Loop->getBeginLoc(),
                                             Loop->getEndLoc()),
             Replacement);
}

} // namespace clang::tidy::misc
