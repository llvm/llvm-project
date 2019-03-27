//===--- TimeSubtractionCheck.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TimeSubtractionCheck.h"
#include "DurationRewriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

// Returns `true` if `Range` is inside a macro definition.
static bool InsideMacroDefinition(const MatchFinder::MatchResult &Result,
                                  SourceRange Range) {
  return !clang::Lexer::makeFileCharRange(
              clang::CharSourceRange::getCharRange(Range),
              *Result.SourceManager, Result.Context->getLangOpts())
              .isValid();
}

static bool isConstructorAssignment(const MatchFinder::MatchResult &Result,
                                    const Expr *Node) {
  return selectFirst<const Expr>(
             "e", match(expr(hasParent(materializeTemporaryExpr(hasParent(
                                 cxxConstructExpr(hasParent(exprWithCleanups(
                                     hasParent(varDecl()))))))))
                            .bind("e"),
                        *Node, *Result.Context)) != nullptr;
}

static bool isArgument(const MatchFinder::MatchResult &Result,
                       const Expr *Node) {
  return selectFirst<const Expr>(
             "e",
             match(expr(hasParent(
                            materializeTemporaryExpr(hasParent(cxxConstructExpr(
                                hasParent(callExpr()),
                                unless(hasParent(cxxOperatorCallExpr())))))))
                       .bind("e"),
                   *Node, *Result.Context)) != nullptr;
}

static bool isReturn(const MatchFinder::MatchResult &Result, const Expr *Node) {
  return selectFirst<const Expr>(
             "e", match(expr(hasParent(materializeTemporaryExpr(hasParent(
                                 cxxConstructExpr(hasParent(exprWithCleanups(
                                     hasParent(returnStmt()))))))))
                            .bind("e"),
                        *Node, *Result.Context)) != nullptr;
}

static bool parensRequired(const MatchFinder::MatchResult &Result,
                           const Expr *Node) {
  // TODO: Figure out any more contexts in which we can omit the surrounding
  // parentheses.
  return !(isConstructorAssignment(Result, Node) || isArgument(Result, Node) ||
           isReturn(Result, Node));
}

void TimeSubtractionCheck::emitDiagnostic(const Expr *Node,
                                          llvm::StringRef Replacement) {
  diag(Node->getBeginLoc(), "perform subtraction in the time domain")
      << FixItHint::CreateReplacement(Node->getSourceRange(), Replacement);
}

void TimeSubtractionCheck::registerMatchers(MatchFinder *Finder) {
  for (auto ScaleName :
       {"Hours", "Minutes", "Seconds", "Millis", "Micros", "Nanos"}) {
    std::string TimeInverse = (llvm::Twine("ToUnix") + ScaleName).str();
    llvm::Optional<DurationScale> Scale = getScaleForTimeInverse(TimeInverse);
    assert(Scale && "Unknow scale encountered");

    auto TimeInverseMatcher = callExpr(callee(
        functionDecl(hasName((llvm::Twine("::absl::") + TimeInverse).str()))
            .bind("func_decl")));

    // Match the cases where we know that the result is a 'Duration' and the
    // first argument is a 'Time'. Just knowing the type of the first operand
    // is not sufficient, since the second operand could be either a 'Time' or
    // a 'Duration'. If we know the result is a 'Duration', we can then infer
    // that the second operand must be a 'Time'.
    auto CallMatcher =
        callExpr(
            callee(functionDecl(hasName(getDurationFactoryForScale(*Scale)))),
            hasArgument(0, binaryOperator(hasOperatorName("-"),
                                          hasLHS(TimeInverseMatcher))
                               .bind("binop")))
            .bind("outer_call");
    Finder->addMatcher(CallMatcher, this);

    // Match cases where we know the second operand is a 'Time'. Since
    // subtracting a 'Time' from a 'Duration' is not defined, in these cases,
    // we always know the first operand is a 'Time' if the second is a 'Time'.
    auto OperandMatcher =
        binaryOperator(hasOperatorName("-"), hasRHS(TimeInverseMatcher))
            .bind("binop");
    Finder->addMatcher(OperandMatcher, this);
  }
}

void TimeSubtractionCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binop");
  std::string InverseName =
      Result.Nodes.getNodeAs<FunctionDecl>("func_decl")->getNameAsString();
  if (InsideMacroDefinition(Result, BinOp->getSourceRange()))
    return;

  llvm::Optional<DurationScale> Scale = getScaleForTimeInverse(InverseName);
  if (!Scale)
    return;

  const auto *OuterCall = Result.Nodes.getNodeAs<CallExpr>("outer_call");
  if (OuterCall) {
    if (InsideMacroDefinition(Result, OuterCall->getSourceRange()))
      return;

    // We're working with the first case of matcher, and need to replace the
    // entire 'Duration' factory call. (Which also means being careful about
    // our order-of-operations and optionally putting in some parenthesis.
    bool NeedParens = parensRequired(Result, OuterCall);

    emitDiagnostic(
        OuterCall,
        (llvm::Twine(NeedParens ? "(" : "") +
         rewriteExprFromNumberToTime(Result, *Scale, BinOp->getLHS()) + " - " +
         rewriteExprFromNumberToTime(Result, *Scale, BinOp->getRHS()) +
         (NeedParens ? ")" : ""))
            .str());
  } else {
    // We're working with the second case of matcher, and either just need to
    // change the arguments, or perhaps remove an outer function call. In the
    // latter case (addressed first), we also need to worry about parenthesis.
    const auto *MaybeCallArg = selectFirst<const CallExpr>(
        "arg", match(expr(hasAncestor(
                         callExpr(callee(functionDecl(hasName(
                                      getDurationFactoryForScale(*Scale)))))
                             .bind("arg"))),
                     *BinOp, *Result.Context));
    if (MaybeCallArg && MaybeCallArg->getArg(0)->IgnoreImpCasts() == BinOp &&
        !InsideMacroDefinition(Result, MaybeCallArg->getSourceRange())) {
      // Handle the case where the matched expression is inside a call which
      // converts it from the inverse to a Duration.  In this case, we replace
      // the outer with just the subtraction expresison, which gives the right
      // type and scale, taking care again about parenthesis.
      bool NeedParens = parensRequired(Result, MaybeCallArg);

      emitDiagnostic(
          MaybeCallArg,
          (llvm::Twine(NeedParens ? "(" : "") +
           rewriteExprFromNumberToTime(Result, *Scale, BinOp->getLHS()) +
           " - " +
           rewriteExprFromNumberToTime(Result, *Scale, BinOp->getRHS()) +
           (NeedParens ? ")" : ""))
              .str());
    } else {
      // In the last case, just convert the arguments and wrap the result in
      // the correct inverse function.
      emitDiagnostic(
          BinOp,
          (llvm::Twine(
               getDurationInverseForScale(*Scale).second.str().substr(2)) +
           "(" + rewriteExprFromNumberToTime(Result, *Scale, BinOp->getLHS()) +
           " - " +
           rewriteExprFromNumberToTime(Result, *Scale, BinOp->getRHS()) + ")")
              .str());
    }
  }
}

} // namespace abseil
} // namespace tidy
} // namespace clang
