//===--- CleanupStaticCastCheck.cpp - clang-tidy-----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CleanupStaticCastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {


std::string getText(const clang::Expr *E, const clang::ASTContext &Context) {
  auto &SM = Context.getSourceManager();
  auto Range = clang::CharSourceRange::getTokenRange(E->getSourceRange());
  return clang::Lexer::getSourceText(Range, SM, Context.getLangOpts()).str();
}

void CleanupStaticCastCheck::registerMatchers(MatchFinder *Finder) {
  // Match static_cast expressions not in templates
  Finder->addMatcher(
      cxxStaticCastExpr(
          unless(hasAncestor(functionTemplateDecl())),
          unless(hasAncestor(classTemplateDecl())),
          unless(isInTemplateInstantiation()))
      .bind("cast"),
      this);
}

void CleanupStaticCastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Cast = Result.Nodes.getNodeAs<CXXStaticCastExpr>("cast");
  if (!Cast)
    return;

  const Expr *SubExpr = Cast->getSubExpr()->IgnoreParenImpCasts();
  QualType SourceType = SubExpr->getType();
  QualType TargetType = Cast->getType();

  // Skip if either type is dependent
  if (SourceType->isDependentType() || TargetType->isDependentType())
    return;

  // Compare canonical types and qualifiers
  SourceType = SourceType.getCanonicalType();
  TargetType = TargetType.getCanonicalType();
  
  if (SourceType == TargetType) {
    auto Diag = 
        diag(Cast->getBeginLoc(),
             "redundant static_cast to the same type %0")  // Removed single quotes
        << TargetType;

    std::string ReplacementText = getText(SubExpr, *Result.Context);
    
    Diag << FixItHint::CreateReplacement(
        Cast->getSourceRange(),
        ReplacementText);
  }
}

} // namespace clang::tidy::modernize