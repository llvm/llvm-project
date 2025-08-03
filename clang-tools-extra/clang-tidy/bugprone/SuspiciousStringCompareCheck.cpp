//===--- SuspiciousStringCompareCheck.cpp - clang-tidy---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuspiciousStringCompareCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

// Semicolon separated list of known string compare-like functions. The list
// must ends with a semicolon.
static const char KnownStringCompareFunctions[] = "__builtin_memcmp;"
                                                  "__builtin_strcasecmp;"
                                                  "__builtin_strcmp;"
                                                  "__builtin_strncasecmp;"
                                                  "__builtin_strncmp;"
                                                  "_mbscmp;"
                                                  "_mbscmp_l;"
                                                  "_mbsicmp;"
                                                  "_mbsicmp_l;"
                                                  "_mbsnbcmp;"
                                                  "_mbsnbcmp_l;"
                                                  "_mbsnbicmp;"
                                                  "_mbsnbicmp_l;"
                                                  "_mbsncmp;"
                                                  "_mbsncmp_l;"
                                                  "_mbsnicmp;"
                                                  "_mbsnicmp_l;"
                                                  "_memicmp;"
                                                  "_memicmp_l;"
                                                  "_stricmp;"
                                                  "_stricmp_l;"
                                                  "_strnicmp;"
                                                  "_strnicmp_l;"
                                                  "_wcsicmp;"
                                                  "_wcsicmp_l;"
                                                  "_wcsnicmp;"
                                                  "_wcsnicmp_l;"
                                                  "lstrcmp;"
                                                  "lstrcmpi;"
                                                  "memcmp;"
                                                  "memicmp;"
                                                  "strcasecmp;"
                                                  "strcmp;"
                                                  "strcmpi;"
                                                  "stricmp;"
                                                  "strncasecmp;"
                                                  "strncmp;"
                                                  "strnicmp;"
                                                  "wcscasecmp;"
                                                  "wcscmp;"
                                                  "wcsicmp;"
                                                  "wcsncmp;"
                                                  "wcsnicmp;"
                                                  "wmemcmp;";

SuspiciousStringCompareCheck::SuspiciousStringCompareCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WarnOnImplicitComparison(Options.get("WarnOnImplicitComparison", true)),
      WarnOnLogicalNotComparison(
          Options.get("WarnOnLogicalNotComparison", false)),
      StringCompareLikeFunctions(
          Options.get("StringCompareLikeFunctions", "")) {}

void SuspiciousStringCompareCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WarnOnImplicitComparison", WarnOnImplicitComparison);
  Options.store(Opts, "WarnOnLogicalNotComparison", WarnOnLogicalNotComparison);
  Options.store(Opts, "StringCompareLikeFunctions", StringCompareLikeFunctions);
}

void SuspiciousStringCompareCheck::registerMatchers(MatchFinder *Finder) {
  // Match relational operators.
  const auto ComparisonUnaryOperator = unaryOperator(hasOperatorName("!"));
  const auto ComparisonBinaryOperator = binaryOperator(isComparisonOperator());
  const auto ComparisonOperator =
      expr(anyOf(ComparisonUnaryOperator, ComparisonBinaryOperator));

  // Add the list of known string compare-like functions and add user-defined
  // functions.
  std::vector<StringRef> FunctionNames = utils::options::parseListPair(
      KnownStringCompareFunctions, StringCompareLikeFunctions);

  // Match a call to a string compare functions.
  const auto FunctionCompareDecl =
      functionDecl(hasAnyName(FunctionNames)).bind("decl");
  const auto DirectStringCompareCallExpr =
      callExpr(hasDeclaration(FunctionCompareDecl)).bind("call");
  const auto MacroStringCompareCallExpr = conditionalOperator(anyOf(
      hasTrueExpression(ignoringParenImpCasts(DirectStringCompareCallExpr)),
      hasFalseExpression(ignoringParenImpCasts(DirectStringCompareCallExpr))));
  // The implicit cast is not present in C.
  const auto StringCompareCallExpr = ignoringParenImpCasts(
      anyOf(DirectStringCompareCallExpr, MacroStringCompareCallExpr));

  if (WarnOnImplicitComparison) {
    // Detect suspicious calls to string compare:
    //     'if (strcmp())'  ->  'if (strcmp() != 0)'
    Finder->addMatcher(
        stmt(anyOf(mapAnyOf(ifStmt, whileStmt, doStmt, forStmt)
                       .with(hasCondition(StringCompareCallExpr)),
                   binaryOperator(hasAnyOperatorName("&&", "||"),
                                  hasEitherOperand(StringCompareCallExpr))))
            .bind("missing-comparison"),
        this);
  }

  if (WarnOnLogicalNotComparison) {
    // Detect suspicious calls to string compared with '!' operator:
    //     'if (!strcmp())'  ->  'if (strcmp() == 0)'
    Finder->addMatcher(unaryOperator(hasOperatorName("!"),
                                     hasUnaryOperand(ignoringParenImpCasts(
                                         StringCompareCallExpr)))
                           .bind("logical-not-comparison"),
                       this);
  }

  // Detect suspicious cast to an inconsistent type (i.e. not integer type).
  Finder->addMatcher(
      traverse(TK_AsIs,
               implicitCastExpr(unless(hasType(isInteger())),
                                hasSourceExpression(StringCompareCallExpr))
                   .bind("invalid-conversion")),
      this);

  // Detect suspicious operator with string compare function as operand.
  Finder->addMatcher(
      binaryOperator(unless(anyOf(isComparisonOperator(), hasOperatorName("&&"),
                                  hasOperatorName("||"), hasOperatorName("="))),
                     hasEitherOperand(StringCompareCallExpr))
          .bind("suspicious-operator"),
      this);

  // Detect comparison to invalid constant: 'strcmp() == -1'.
  const auto InvalidLiteral = ignoringParenImpCasts(
      anyOf(integerLiteral(unless(equals(0))),
            unaryOperator(
                hasOperatorName("-"),
                has(ignoringParenImpCasts(integerLiteral(unless(equals(0)))))),
            characterLiteral(), cxxBoolLiteral()));

  Finder->addMatcher(
      binaryOperator(isComparisonOperator(),
                     hasOperands(StringCompareCallExpr, InvalidLiteral))
          .bind("invalid-comparison"),
      this);
}

void SuspiciousStringCompareCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Decl = Result.Nodes.getNodeAs<FunctionDecl>("decl");
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  assert(Decl != nullptr && Call != nullptr);

  if (Result.Nodes.getNodeAs<Stmt>("missing-comparison")) {
    SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        Call->getRParenLoc(), 0, Result.Context->getSourceManager(),
        getLangOpts());

    diag(Call->getBeginLoc(),
         "function %0 is called without explicitly comparing result")
        << Decl << FixItHint::CreateInsertion(EndLoc, " != 0");
  }

  if (const auto *E = Result.Nodes.getNodeAs<Expr>("logical-not-comparison")) {
    SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        Call->getRParenLoc(), 0, Result.Context->getSourceManager(),
        getLangOpts());
    SourceLocation NotLoc = E->getBeginLoc();

    diag(Call->getBeginLoc(),
         "function %0 is compared using logical not operator")
        << Decl
        << FixItHint::CreateRemoval(
               CharSourceRange::getTokenRange(NotLoc, NotLoc))
        << FixItHint::CreateInsertion(EndLoc, " == 0");
  }

  if (Result.Nodes.getNodeAs<Stmt>("invalid-comparison")) {
    diag(Call->getBeginLoc(),
         "function %0 is compared to a suspicious constant")
        << Decl;
  }

  if (const auto *BinOp =
          Result.Nodes.getNodeAs<BinaryOperator>("suspicious-operator")) {
    diag(Call->getBeginLoc(), "results of function %0 used by operator '%1'")
        << Decl << BinOp->getOpcodeStr();
  }

  if (Result.Nodes.getNodeAs<Stmt>("invalid-conversion")) {
    diag(Call->getBeginLoc(), "function %0 has suspicious implicit cast")
        << Decl;
  }
}

} // namespace clang::tidy::bugprone
