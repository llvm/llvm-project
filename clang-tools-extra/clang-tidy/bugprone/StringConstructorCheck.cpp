//===--- StringConstructorCheck.cpp - clang-tidy---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "StringConstructorCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

namespace {
AST_MATCHER_P(IntegerLiteral, isBiggerThan, unsigned, N) {
  return Node.getValue().getZExtValue() > N;
}
} // namespace

StringConstructorCheck::StringConstructorCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WarnOnLargeLength(Options.get("WarnOnLargeLength", 1) != 0),
      LargeLengthThreshold(Options.get("LargeLengthThreshold", 0x800000)) {}

void StringConstructorCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WarnOnLargeLength", WarnOnLargeLength);
  Options.store(Opts, "LargeLengthThreshold", LargeLengthThreshold);
}

void StringConstructorCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  const auto ZeroExpr = expr(ignoringParenImpCasts(integerLiteral(equals(0))));
  const auto CharExpr = expr(ignoringParenImpCasts(characterLiteral()));
  const auto NegativeExpr = expr(ignoringParenImpCasts(
      unaryOperator(hasOperatorName("-"),
                    hasUnaryOperand(integerLiteral(unless(equals(0)))))));
  const auto LargeLengthExpr = expr(ignoringParenImpCasts(
      integerLiteral(isBiggerThan(LargeLengthThreshold))));
  const auto CharPtrType = type(anyOf(pointerType(), arrayType()));

  // Match a string-literal; even through a declaration with initializer.
  const auto BoundStringLiteral = stringLiteral().bind("str");
  const auto ConstStrLiteralDecl = varDecl(
      isDefinition(), hasType(constantArrayType()), hasType(isConstQualified()),
      hasInitializer(ignoringParenImpCasts(BoundStringLiteral)));
  const auto ConstPtrStrLiteralDecl = varDecl(
      isDefinition(),
      hasType(pointerType(pointee(isAnyCharacter(), isConstQualified()))),
      hasInitializer(ignoringParenImpCasts(BoundStringLiteral)));
  const auto ConstStrLiteral = expr(ignoringParenImpCasts(anyOf(
      BoundStringLiteral, declRefExpr(hasDeclaration(anyOf(
                              ConstPtrStrLiteralDecl, ConstStrLiteralDecl))))));

  // Check the fill constructor. Fills the string with n consecutive copies of
  // character c. [i.e string(size_t n, char c);].
  Finder->addMatcher(
      cxxConstructExpr(
          hasDeclaration(cxxMethodDecl(hasName("basic_string"))),
          hasArgument(0, hasType(qualType(isInteger()))),
          hasArgument(1, hasType(qualType(isInteger()))),
          anyOf(
              // Detect the expression: string('x', 40);
              hasArgument(0, CharExpr.bind("swapped-parameter")),
              // Detect the expression: string(0, ...);
              hasArgument(0, ZeroExpr.bind("empty-string")),
              // Detect the expression: string(-4, ...);
              hasArgument(0, NegativeExpr.bind("negative-length")),
              // Detect the expression: string(0x1234567, ...);
              hasArgument(0, LargeLengthExpr.bind("large-length"))))
          .bind("constructor"),
      this);

  // Check the literal string constructor with char pointer and length
  // parameters. [i.e. string (const char* s, size_t n);]
  Finder->addMatcher(
      cxxConstructExpr(
          hasDeclaration(cxxMethodDecl(hasName("basic_string"))),
          hasArgument(0, hasType(CharPtrType)),
          hasArgument(1, hasType(isInteger())),
          anyOf(
              // Detect the expression: string("...", 0);
              hasArgument(1, ZeroExpr.bind("empty-string")),
              // Detect the expression: string("...", -4);
              hasArgument(1, NegativeExpr.bind("negative-length")),
              // Detect the expression: string("lit", 0x1234567);
              hasArgument(1, LargeLengthExpr.bind("large-length")),
              // Detect the expression: string("lit", 5)
              allOf(hasArgument(0, ConstStrLiteral.bind("literal-with-length")),
                    hasArgument(1, ignoringParenImpCasts(
                                       integerLiteral().bind("int"))))))
          .bind("constructor"),
      this);
}

void StringConstructorCheck::check(const MatchFinder::MatchResult &Result) {
  const ASTContext &Ctx = *Result.Context;
  const auto *E = Result.Nodes.getNodeAs<CXXConstructExpr>("constructor");
  assert(E && "missing constructor expression");
  SourceLocation Loc = E->getBeginLoc();

  if (Result.Nodes.getNodeAs<Expr>("swapped-parameter")) {
    const Expr *P0 = E->getArg(0);
    const Expr *P1 = E->getArg(1);
    diag(Loc, "string constructor parameters are probably swapped;"
              " expecting string(count, character)")
        << tooling::fixit::createReplacement(*P0, *P1, Ctx)
        << tooling::fixit::createReplacement(*P1, *P0, Ctx);
  } else if (Result.Nodes.getNodeAs<Expr>("empty-string")) {
    diag(Loc, "constructor creating an empty string");
  } else if (Result.Nodes.getNodeAs<Expr>("negative-length")) {
    diag(Loc, "negative value used as length parameter");
  } else if (Result.Nodes.getNodeAs<Expr>("large-length")) {
    if (WarnOnLargeLength)
      diag(Loc, "suspicious large length parameter");
  } else if (Result.Nodes.getNodeAs<Expr>("literal-with-length")) {
    const auto *Str = Result.Nodes.getNodeAs<StringLiteral>("str");
    const auto *Lit = Result.Nodes.getNodeAs<IntegerLiteral>("int");
    if (Lit->getValue().ugt(Str->getLength())) {
      diag(Loc, "length is bigger then string literal size");
    }
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
