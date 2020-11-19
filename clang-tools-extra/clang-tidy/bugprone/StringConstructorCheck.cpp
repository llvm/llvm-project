//===--- StringConstructorCheck.cpp - clang-tidy---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StringConstructorCheck.h"
#include "../utils/OptionsUtils.h"
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

const char DefaultStringNames[] =
    "::std::basic_string;::std::basic_string_view";

static std::vector<StringRef>
removeNamespaces(const std::vector<std::string> &Names) {
  std::vector<StringRef> Result;
  Result.reserve(Names.size());
  for (StringRef Name : Names) {
    std::string::size_type ColonPos = Name.rfind(':');
    Result.push_back(
        Name.substr(ColonPos == std::string::npos ? 0 : ColonPos + 1));
  }
  return Result;
}

} // namespace

StringConstructorCheck::StringConstructorCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WarnOnLargeLength(Options.get("WarnOnLargeLength", true)),
      LargeLengthThreshold(Options.get("LargeLengthThreshold", 0x800000)),
      StringNames(utils::options::parseStringList(
          Options.get("StringNames", DefaultStringNames))) {}

void StringConstructorCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WarnOnLargeLength", WarnOnLargeLength);
  Options.store(Opts, "LargeLengthThreshold", LargeLengthThreshold);
  Options.store(Opts, "StringNames", DefaultStringNames);
}

void StringConstructorCheck::registerMatchers(MatchFinder *Finder) {
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
          hasDeclaration(cxxConstructorDecl(ofClass(
              cxxRecordDecl(hasAnyName(removeNamespaces(StringNames)))))),
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

  // Check the literal string constructor with char pointer.
  // [i.e. string (const char* s);]
  Finder->addMatcher(
      traverse(TK_AsIs,
               cxxConstructExpr(
                   hasDeclaration(cxxConstructorDecl(ofClass(cxxRecordDecl(
                       hasAnyName(removeNamespaces(StringNames)))))),
                   hasArgument(0, expr().bind("from-ptr")),
                   // do not match std::string(ptr, int)
                   // match std::string(ptr, alloc)
                   // match std::string(ptr)
                   anyOf(hasArgument(1, unless(hasType(isInteger()))),
                         argumentCountIs(1)))
                   .bind("constructor")),
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
      diag(Loc, "length is bigger than string literal size");
    }
  } else if (const auto *Ptr = Result.Nodes.getNodeAs<Expr>("from-ptr")) {
    Expr::EvalResult ConstPtr;
    if (!Ptr->isInstantiationDependent() &&
        Ptr->EvaluateAsRValue(ConstPtr, Ctx) &&
        ((ConstPtr.Val.isInt() && ConstPtr.Val.getInt().isNullValue()) ||
         (ConstPtr.Val.isLValue() && ConstPtr.Val.isNullPointer()))) {
      diag(Loc, "constructing string from nullptr is undefined behaviour");
    }
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
