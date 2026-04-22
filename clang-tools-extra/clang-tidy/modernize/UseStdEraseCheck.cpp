//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStdEraseCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringRef.h"
#include <initializer_list>
#include <string_view>

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {

constexpr std::array<llvm::StringRef, 2> EraseEndMethodNames = {"end", "cend"};
constexpr std::array<llvm::StringRef, 2> EraseEndFreeNames = {"end", "cend"};
constexpr const char *EraseThis = "EraseThis";

AST_MATCHER(Expr, hasSideEffects) {
  return Node.HasSideEffects(Finder->getASTContext());
}

auto makeExprMatcher(
    const ast_matchers::internal::Matcher<Expr> &ArgumentMatcher,
    ArrayRef<StringRef> MethodNames, ArrayRef<StringRef> FreeNames) {
  return expr(
      anyOf(cxxMemberCallExpr(argumentCountIs(0),
                              callee(cxxMethodDecl(hasAnyName(MethodNames))),
                              on(ArgumentMatcher)),
            callExpr(argumentCountIs(1), hasArgument(0, ArgumentMatcher),
                     hasDeclaration(functionDecl(hasAnyName(FreeNames))))));
}

ast_matchers::internal::Matcher<Expr> makeMatcherPair() {
  ast_matchers::internal::Matcher<CallExpr> ArgumentMatcher = allOf(
      hasArgument(
          0, makeExprMatcher(expr(unless(hasSideEffects())).bind(EraseThis),
                             {"begin"}, {"::std::begin"})),
      hasArgument(
          1, makeExprMatcher(
                 expr(matchers::isStatementIdenticalToBoundNode(EraseThis)),
                 {"end"}, {"::std::end"})),
      hasArgument(2, expr().bind("valueOrCond")));

  return callExpr(callee(functionDecl(hasAnyName("remove", "remove_if"))),
                  argumentCountIs(3), ArgumentMatcher)
      .bind("remove");
}

} // namespace

void UseStdEraseCheck::registerMatchers(MatchFinder *Finder) {
  const auto IsCpp20EraseContainer = cxxRecordDecl(
      hasAnyName("vector", "deque", "list", "forward_list", "basic_string"),
      isInStdNamespace());

  const auto EraseableContainerType = type(hasUnqualifiedDesugaredType(
      tagType(hasDeclaration(IsCpp20EraseContainer))));

  auto EraseEndCheck = makeExprMatcher(
      expr(matchers::isStatementIdenticalToBoundNode(EraseThis)),
      EraseEndMethodNames, EraseEndFreeNames);

  Finder->addMatcher(
      cxxMemberCallExpr(callee(cxxMethodDecl(hasName("erase"))),
                        hasArgument(0, makeMatcherPair()),
                        hasArgument(1, EraseEndCheck),
                        on(anyOf(hasType(EraseableContainerType),
                                 hasType(pointsTo(EraseableContainerType)))))
          .bind("erase"),
      this);
}

void UseStdEraseCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *EraseCall = Result.Nodes.getNodeAs<CXXMemberCallExpr>("erase");
  const auto *RemoveCall = Result.Nodes.getNodeAs<CallExpr>("remove");
  const auto *ContainerThis = Result.Nodes.getNodeAs<Expr>(EraseThis);
  const auto *ValueOrCond = Result.Nodes.getNodeAs<Expr>("valueOrCond");

  if (!EraseCall || !RemoveCall || !ContainerThis || !ValueOrCond)
    return;

  const CXXMethodDecl *EraseMethod = EraseCall->getMethodDecl();
  if (!EraseMethod)
    return;

  const std::string RemoveFuncName =
      RemoveCall->getDirectCallee()->getName().str();

  const std::string ReplacementFreeFunc =
      RemoveFuncName == "remove" ? "std::erase" : "std::erase_if";

  std::string Replacement =
      ReplacementFreeFunc + "(" +
      Lexer::getSourceText(
          CharSourceRange::getTokenRange(ContainerThis->getSourceRange()),
          Result.Context->getSourceManager(), Result.Context->getLangOpts())
          .str() +
      ", " +
      Lexer::getSourceText(
          CharSourceRange::getTokenRange(ValueOrCond->getSourceRange()),
          Result.Context->getSourceManager(), Result.Context->getLangOpts())
          .str() +
      ")";

  diag(EraseCall->getExprLoc(),
       "prefer %0 over the erase-" + RemoveFuncName + " idiom")
      << ReplacementFreeFunc
      << FixItHint::CreateReplacement(EraseCall->getSourceRange(), Replacement);
}

} // namespace clang::tidy::modernize
