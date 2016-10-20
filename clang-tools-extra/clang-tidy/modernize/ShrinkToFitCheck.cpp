//===--- ShrinkToFitCheck.cpp - clang-tidy---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ShrinkToFitCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringRef.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

void ShrinkToFitCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus11)
    return;

  // Swap as a function need not to be considered, because rvalue can not
  // be bound to a non-const reference.
  const auto ShrinkableAsMember =
      memberExpr(member(valueDecl().bind("ContainerDecl")));
  const auto ShrinkableAsDecl =
      declRefExpr(hasDeclaration(valueDecl().bind("ContainerDecl")));
  const auto CopyCtorCall = cxxConstructExpr(hasArgument(
      0, anyOf(ShrinkableAsMember, ShrinkableAsDecl,
               unaryOperator(has(ignoringParenImpCasts(ShrinkableAsMember))),
               unaryOperator(has(ignoringParenImpCasts(ShrinkableAsDecl))))));
  const auto SwapParam =
      expr(anyOf(memberExpr(member(equalsBoundNode("ContainerDecl"))),
                 declRefExpr(hasDeclaration(equalsBoundNode("ContainerDecl"))),
                 unaryOperator(has(ignoringParenImpCasts(
                     memberExpr(member(equalsBoundNode("ContainerDecl")))))),
                 unaryOperator(has(ignoringParenImpCasts(declRefExpr(
                     hasDeclaration(equalsBoundNode("ContainerDecl"))))))));

  Finder->addMatcher(
      cxxMemberCallExpr(
          on(hasType(namedDecl(
              hasAnyName("std::basic_string", "std::deque", "std::vector")))),
          callee(cxxMethodDecl(hasName("swap"))),
          has(ignoringParenImpCasts(memberExpr(hasDescendant(CopyCtorCall)))),
          hasArgument(0, SwapParam.bind("ContainerToShrink")),
          unless(isInTemplateInstantiation()))
          .bind("CopyAndSwapTrick"),
      this);
}

void ShrinkToFitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MemberCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("CopyAndSwapTrick");
  const auto *Container = Result.Nodes.getNodeAs<Expr>("ContainerToShrink");
  FixItHint Hint;

  if (!MemberCall->getLocStart().isMacroID()) {
    const LangOptions &Opts = getLangOpts();
    std::string ReplacementText;
    if (const auto *UnaryOp = llvm::dyn_cast<UnaryOperator>(Container)) {
      ReplacementText =
          Lexer::getSourceText(CharSourceRange::getTokenRange(
                                   UnaryOp->getSubExpr()->getSourceRange()),
                               *Result.SourceManager, Opts);
      ReplacementText += "->shrink_to_fit()";
    } else {
      ReplacementText = Lexer::getSourceText(
          CharSourceRange::getTokenRange(Container->getSourceRange()),
          *Result.SourceManager, Opts);
      ReplacementText += ".shrink_to_fit()";
    }

    Hint = FixItHint::CreateReplacement(MemberCall->getSourceRange(),
                                        ReplacementText);
  }

  diag(MemberCall->getLocStart(), "the shrink_to_fit method should be used "
                                  "to reduce the capacity of a shrinkable "
                                  "container")
      << Hint;
}

} // namespace modernize
} // namespace tidy
} // namespace clang
