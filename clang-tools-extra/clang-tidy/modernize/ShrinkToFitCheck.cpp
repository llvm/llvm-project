//===--- ShrinkToFitCheck.cpp - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ShrinkToFitCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringRef.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

void ShrinkToFitCheck::registerMatchers(MatchFinder *Finder) {
  // Swap as a function need not to be considered, because rvalue can not
  // be bound to a non-const reference.
  const auto ShrinkableExpr = mapAnyOf(memberExpr, declRefExpr);
  const auto Shrinkable =
      ShrinkableExpr.with(hasDeclaration(valueDecl().bind("ContainerDecl")));
  const auto BoundShrinkable = ShrinkableExpr.with(
      hasDeclaration(valueDecl(equalsBoundNode("ContainerDecl"))));

  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(cxxMethodDecl(hasName("swap"))),
          hasArgument(
              0, anyOf(Shrinkable, unaryOperator(hasUnaryOperand(Shrinkable)))),
          on(cxxConstructExpr(hasArgument(
              0,
              expr(anyOf(BoundShrinkable,
                         unaryOperator(hasUnaryOperand(BoundShrinkable))),
                   hasType(hasCanonicalType(hasDeclaration(namedDecl(hasAnyName(
                       "std::basic_string", "std::deque", "std::vector"))))))
                  .bind("ContainerToShrink")))))
          .bind("CopyAndSwapTrick"),
      this);
}

void ShrinkToFitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MemberCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("CopyAndSwapTrick");
  const auto *Container = Result.Nodes.getNodeAs<Expr>("ContainerToShrink");
  FixItHint Hint;

  if (!MemberCall->getBeginLoc().isMacroID()) {
    const LangOptions &Opts = getLangOpts();
    std::string ReplacementText;
    if (const auto *UnaryOp = llvm::dyn_cast<UnaryOperator>(Container)) {
      ReplacementText = std::string(
          Lexer::getSourceText(CharSourceRange::getTokenRange(
                                   UnaryOp->getSubExpr()->getSourceRange()),
                               *Result.SourceManager, Opts));
      ReplacementText += "->shrink_to_fit()";
    } else {
      ReplacementText = std::string(Lexer::getSourceText(
          CharSourceRange::getTokenRange(Container->getSourceRange()),
          *Result.SourceManager, Opts));
      ReplacementText += ".shrink_to_fit()";
    }

    Hint = FixItHint::CreateReplacement(MemberCall->getSourceRange(),
                                        ReplacementText);
  }

  diag(MemberCall->getBeginLoc(), "the shrink_to_fit method should be used "
                                  "to reduce the capacity of a shrinkable "
                                  "container")
      << Hint;
}

} // namespace clang::tidy::modernize
