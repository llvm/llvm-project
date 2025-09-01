//===--- PreferIsaOrDynCastInConditionalsCheck.cpp - clang-tidy
//---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PreferIsaOrDynCastInConditionalsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

namespace {
AST_MATCHER(Expr, isMacroID) { return Node.getExprLoc().isMacroID(); }
} // namespace

void PreferIsaOrDynCastInConditionalsCheck::registerMatchers(
    MatchFinder *Finder) {
  auto Condition = hasCondition(implicitCastExpr(has(
      callExpr(unless(isMacroID()), unless(cxxMemberCallExpr()),
               anyOf(callee(namedDecl(hasName("cast"))),
                     callee(namedDecl(hasName("dyn_cast")).bind("dyn_cast"))))
          .bind("call"))));

  auto Any = anyOf(
      has(declStmt(containsDeclaration(
          0, varDecl(hasInitializer(callExpr(unless(isMacroID()),
                                             unless(cxxMemberCallExpr()),
                                             callee(namedDecl(hasName("cast"))))
                                        .bind("assign")))))),
      Condition);

  auto CallExpression =
      callExpr(

          unless(isMacroID()), unless(cxxMemberCallExpr()),
          callee(namedDecl(hasAnyName("isa", "cast", "cast_or_null", "dyn_cast",
                                      "dyn_cast_or_null"))
                     .bind("func")),
          hasArgument(0, mapAnyOf(declRefExpr, cxxMemberCallExpr).bind("arg")))
          .bind("rhs");

  Finder->addMatcher(
      traverse(
          TK_AsIs,
          stmt(anyOf(
              ifStmt(Any), whileStmt(Any), doStmt(Condition),
              binaryOperator(unless(isExpansionInFileMatching(
                                 "llvm/include/llvm/Support/Casting.h")),
                             hasOperatorName("&&"),
                             hasLHS(implicitCastExpr().bind("lhs")),
                             hasRHS(anyOf(implicitCastExpr(has(CallExpression)),
                                          CallExpression)))
                  .bind("and")))),
      this);
}

void PreferIsaOrDynCastInConditionalsCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *MatchedDecl = Result.Nodes.getNodeAs<CallExpr>("assign")) {
    SourceLocation StartLoc = MatchedDecl->getCallee()->getExprLoc();
    SourceLocation EndLoc =
        StartLoc.getLocWithOffset(StringRef("cast").size() - 1);

    diag(MatchedDecl->getBeginLoc(),
         "cast<> in conditional will assert rather than return a null pointer")
        << FixItHint::CreateReplacement(SourceRange(StartLoc, EndLoc),
                                        "dyn_cast");
  } else if (const auto *MatchedDecl =
                 Result.Nodes.getNodeAs<CallExpr>("call")) {
    SourceLocation StartLoc = MatchedDecl->getCallee()->getExprLoc();
    SourceLocation EndLoc =
        StartLoc.getLocWithOffset(StringRef("cast").size() - 1);

    StringRef Message =
        "cast<> in conditional will assert rather than return a null pointer";
    if (Result.Nodes.getNodeAs<NamedDecl>("dyn_cast"))
      Message = "return value from dyn_cast<> not used";

    diag(MatchedDecl->getBeginLoc(), Message)
        << FixItHint::CreateReplacement(SourceRange(StartLoc, EndLoc), "isa");
  } else if (const auto *MatchedDecl =
                 Result.Nodes.getNodeAs<BinaryOperator>("and")) {
    const auto *LHS = Result.Nodes.getNodeAs<ImplicitCastExpr>("lhs");
    const auto *RHS = Result.Nodes.getNodeAs<CallExpr>("rhs");
    const auto *Arg = Result.Nodes.getNodeAs<Expr>("arg");
    const auto *Func = Result.Nodes.getNodeAs<NamedDecl>("func");

    assert(LHS && "LHS is null");
    assert(RHS && "RHS is null");
    assert(Arg && "Arg is null");
    assert(Func && "Func is null");

    StringRef LHSString(Lexer::getSourceText(
        CharSourceRange::getTokenRange(LHS->getSourceRange()),
        *Result.SourceManager, getLangOpts()));

    StringRef ArgString(Lexer::getSourceText(
        CharSourceRange::getTokenRange(Arg->getSourceRange()),
        *Result.SourceManager, getLangOpts()));

    if (ArgString != LHSString)
      return;

    StringRef RHSString(Lexer::getSourceText(
        CharSourceRange::getTokenRange(RHS->getSourceRange()),
        *Result.SourceManager, getLangOpts()));

    std::string Replacement("isa_and_nonnull");
    Replacement += RHSString.substr(Func->getName().size());

    diag(MatchedDecl->getBeginLoc(),
         "isa_and_nonnull<> is preferred over an explicit test for null "
         "followed by calling isa<>")
        << FixItHint::CreateReplacement(SourceRange(MatchedDecl->getBeginLoc(),
                                                    MatchedDecl->getEndLoc()),
                                        Replacement);
  }
}

} // namespace clang::tidy::llvm_check
