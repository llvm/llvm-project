//===--- RedundantControlFlowCheck.cpp - clang-tidy------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RedundantControlFlowCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {

const char *const RedundantReturnDiag = "redundant return statement at the end "
                                        "of a function with a void return type";
const char *const RedundantContinueDiag = "redundant continue statement at the "
                                          "end of loop statement";

bool isLocationInMacroExpansion(const SourceManager &SM, SourceLocation Loc) {
  return SM.isMacroBodyExpansion(Loc) || SM.isMacroArgExpansion(Loc);
}

} // namespace

void RedundantControlFlowCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(isDefinition(), returns(voidType()),
                   has(compoundStmt(hasAnySubstatement(returnStmt(
                                        unless(has(expr()))))).bind("return"))),
      this);
  auto CompoundContinue =
      has(compoundStmt(hasAnySubstatement(continueStmt())).bind("continue"));
  Finder->addMatcher(
      stmt(anyOf(forStmt(), cxxForRangeStmt(), whileStmt(), doStmt()),
           CompoundContinue),
      this);
}

void RedundantControlFlowCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Return = Result.Nodes.getNodeAs<CompoundStmt>("return"))
    checkRedundantReturn(Result, Return);
  else if (const auto *Continue =
               Result.Nodes.getNodeAs<CompoundStmt>("continue"))
    checkRedundantContinue(Result, Continue);
}

void RedundantControlFlowCheck::checkRedundantReturn(
    const MatchFinder::MatchResult &Result, const CompoundStmt *Block) {
  CompoundStmt::const_reverse_body_iterator last = Block->body_rbegin();
  if (const auto *Return = dyn_cast<ReturnStmt>(*last))
    issueDiagnostic(Result, Block, Return->getSourceRange(),
                    RedundantReturnDiag);
}

void RedundantControlFlowCheck::checkRedundantContinue(
    const MatchFinder::MatchResult &Result, const CompoundStmt *Block) {
  CompoundStmt::const_reverse_body_iterator last = Block->body_rbegin();
  if (const auto *Continue = dyn_cast<ContinueStmt>(*last))
    issueDiagnostic(Result, Block, Continue->getSourceRange(),
                    RedundantContinueDiag);
}

void RedundantControlFlowCheck::issueDiagnostic(
    const MatchFinder::MatchResult &Result, const CompoundStmt *const Block,
    const SourceRange &StmtRange, const char *const Diag) {
  SourceManager &SM = *Result.SourceManager;
  if (isLocationInMacroExpansion(SM, StmtRange.getBegin()))
    return;

  CompoundStmt::const_reverse_body_iterator Previous = ++Block->body_rbegin();
  SourceLocation Start;
  if (Previous != Block->body_rend())
    Start = Lexer::findLocationAfterToken(
        dyn_cast<Stmt>(*Previous)->getLocEnd(), tok::semi, SM, getLangOpts(),
        /*SkipTrailingWhitespaceAndNewLine=*/true);
  if (!Start.isValid())
    Start = StmtRange.getBegin();
  auto RemovedRange = CharSourceRange::getCharRange(
      Start, Lexer::findLocationAfterToken(
                 StmtRange.getEnd(), tok::semi, SM, getLangOpts(),
                 /*SkipTrailingWhitespaceAndNewLine=*/true));

  diag(StmtRange.getBegin(), Diag) << FixItHint::CreateRemoval(RemovedRange);
}

} // namespace readability
} // namespace tidy
} // namespace clang
