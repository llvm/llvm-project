//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StringviewSubstrCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void StringViewSubstrCheck::registerMatchers(MatchFinder *Finder) {
  // Match string_view type
  const auto StringViewDecl = recordDecl(hasName("::std::basic_string_view"));
  const auto IsStringView = qualType(
      hasUnqualifiedDesugaredType(recordType(hasDeclaration(StringViewDecl))));

  // Length/size matcher reused in multiple places
  const auto LengthMatcher = cxxMemberCallExpr(callee(memberExpr(hasDeclaration(
      cxxMethodDecl(anyOf(hasName("length"), hasName("size")))))));

  // Match various forms of zero
  const auto IsZero =
      expr(anyOf(ignoringParenImpCasts(integerLiteral(equals(0))),
                 ignoringParenImpCasts(declRefExpr(
                     to(varDecl(hasInitializer(integerLiteral(equals(0)))))))));

  // Match substr() call patterns
  const auto SubstrCall =
      cxxMemberCallExpr(
          callee(memberExpr(hasDeclaration(cxxMethodDecl(hasName("substr"))))),
          on(expr(hasType(IsStringView)).bind("source")),
          // substr always has 2 args (second one may be defaulted)
          argumentCountIs(2),
          anyOf(
              // Case 1: sv.substr(n, npos) -> remove_prefix
              allOf(hasArgument(0, expr().bind("prefix_n")),
                    hasArgument(1, cxxDefaultArgExpr())),

              // Case 2: sv.substr(0, sv.length()) or sv.substr(0, sv.length() -
              // 0) -> redundant self-copy
              allOf(hasArgument(0, IsZero.bind("zero")),
                    hasArgument(
                        1, anyOf(LengthMatcher.bind("full_length"),
                                 binaryOperator(
                                     hasOperatorName("-"),
                                     hasLHS(LengthMatcher.bind("full_length")),
                                     hasRHS(IsZero))))),

              // Case 3: sv.substr(0, sv.length() - n) -> remove_suffix
              allOf(hasArgument(0, IsZero),
                    hasArgument(
                        1, binaryOperator(
                               hasOperatorName("-"),
                               hasLHS(LengthMatcher.bind("length_call")),
                               hasRHS(expr(unless(IsZero)).bind("suffix_n")))
                               .bind("length_minus_n")))))
          .bind("substr_call");

  // Only match assignments not part of larger expressions
  Finder->addMatcher(
      stmt(cxxOperatorCallExpr(
               unless(isInTemplateInstantiation()),
               hasOverloadedOperatorName("="),
               hasArgument(0, expr(hasType(IsStringView)).bind("target")),
               hasArgument(1, SubstrCall))
               .bind("assignment"),
           unless(anyOf(hasAncestor(varDecl()), hasAncestor(callExpr())))),
      this);
}

void StringViewSubstrCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Assignment =
      Result.Nodes.getNodeAs<CXXOperatorCallExpr>("assignment");
  const auto *Target = Result.Nodes.getNodeAs<Expr>("target");
  const auto *Source = Result.Nodes.getNodeAs<Expr>("source");
  const auto *SubstrCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("substr_call");
  const auto *PrefixN = Result.Nodes.getNodeAs<Expr>("prefix_n");
  const auto *FullLength = Result.Nodes.getNodeAs<Expr>("full_length");
  const auto *LengthCall = Result.Nodes.getNodeAs<Expr>("length_call");
  const auto *SuffixN = Result.Nodes.getNodeAs<Expr>("suffix_n");

  if (!Assignment || !Target || !Source || !SubstrCall)
    return;

  const auto *TargetDRE = dyn_cast<DeclRefExpr>(Target->IgnoreParenImpCasts());
  const auto *SourceDRE = dyn_cast<DeclRefExpr>(Source->IgnoreParenImpCasts());

  if (!TargetDRE || !SourceDRE)
    return;

  const bool IsSameVar = (TargetDRE->getDecl() == SourceDRE->getDecl());
  const std::string TargetName = TargetDRE->getNameInfo().getAsString();
  const std::string SourceName = SourceDRE->getNameInfo().getAsString();

  // Case 1: remove_prefix
  if (PrefixN) {
    if (!IsSameVar)
      return;

    const std::string PrefixText =
        Lexer::getSourceText(
            CharSourceRange::getTokenRange(PrefixN->getSourceRange()),
            *Result.SourceManager, Result.Context->getLangOpts())
            .str();

    const std::string Replacement =
        TargetName + ".remove_prefix(" + PrefixText + ")";
    diag(Assignment->getBeginLoc(), "prefer 'remove_prefix' over 'substr' for "
                                    "removing characters from the start")
        << FixItHint::CreateReplacement(Assignment->getSourceRange(),
                                        Replacement);
    return;
  }

  // Case 2: redundant full copy
  if (FullLength) {
    const auto *LengthObject =
        cast<CXXMemberCallExpr>(FullLength)->getImplicitObjectArgument();
    const auto *LengthDRE =
        dyn_cast<DeclRefExpr>(LengthObject->IgnoreParenImpCasts());

    if (!LengthDRE || LengthDRE->getDecl() != SourceDRE->getDecl())
      return;

    if (IsSameVar) {
      // Remove redundant self copy including trailing semicolon
      const SourceLocation EndLoc = Lexer::findLocationAfterToken(
          Assignment->getEndLoc(), tok::semi, *Result.SourceManager,
          Result.Context->getLangOpts(), false);

      if (EndLoc.isValid()) {
        diag(Assignment->getBeginLoc(), "redundant self-copy")
            << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
                   Assignment->getBeginLoc(), EndLoc));
      }
    } else {
      // Direct copy between different variables
      const std::string Replacement = TargetName + " = " + SourceName;
      diag(Assignment->getBeginLoc(), "prefer direct copy over substr")
          << FixItHint::CreateReplacement(Assignment->getSourceRange(),
                                          Replacement);
    }
    return;
  }

  // Case 3: remove_suffix
  if (LengthCall && SuffixN) {
    const auto *LengthObject =
        cast<CXXMemberCallExpr>(LengthCall)->getImplicitObjectArgument();
    const auto *LengthDRE =
        dyn_cast<DeclRefExpr>(LengthObject->IgnoreParenImpCasts());

    if (!LengthDRE || LengthDRE->getDecl() != SourceDRE->getDecl())
      return;

    const std::string SuffixText =
        Lexer::getSourceText(
            CharSourceRange::getTokenRange(SuffixN->getSourceRange()),
            *Result.SourceManager, Result.Context->getLangOpts())
            .str();

    if (IsSameVar) {
      const std::string Replacement =
          TargetName + ".remove_suffix(" + SuffixText + ")";
      diag(Assignment->getBeginLoc(), "prefer 'remove_suffix' over 'substr' "
                                      "for removing characters from the end")
          << FixItHint::CreateReplacement(Assignment->getSourceRange(),
                                          Replacement);
    } else {
      const std::string Replacement = TargetName + " = " + SourceName + ";\n" +
                                      "  " + TargetName + ".remove_suffix(" +
                                      SuffixText + ")";

      diag(Assignment->getBeginLoc(),
           "prefer assignment and remove_suffix over substr")
          << FixItHint::CreateReplacement(Assignment->getSourceRange(),
                                          Replacement);
    }

    return;
  }
}

} // namespace clang::tidy::readability
