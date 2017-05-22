//===--- RedundantDeclarationCheck.cpp - clang-tidy------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RedundantDeclarationCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

void RedundantDeclarationCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      namedDecl(
          anyOf(varDecl(unless(isDefinition())),
                functionDecl(unless(anyOf(isDefinition(), isDefaulted())))))
          .bind("Decl"),
      this);
}

void RedundantDeclarationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *D = Result.Nodes.getNodeAs<NamedDecl>("Decl");
  const auto *Prev = D->getPreviousDecl();
  if (!Prev)
    return;
  if (!Prev->getLocation().isValid())
    return;
  if (Prev->getLocation() == D->getLocation())
    return;

  const SourceManager &SM = *Result.SourceManager;

  const bool DifferentHeaders =
      !SM.isInMainFile(D->getLocation()) &&
      !SM.isWrittenInSameFile(Prev->getLocation(), D->getLocation());

  bool MultiVar = false;
  if (const auto *VD = dyn_cast<VarDecl>(D)) {
    // Is this a multivariable declaration?
    for (const auto Other : VD->getDeclContext()->decls()) {
      if (Other != D && Other->getLocStart() == VD->getLocStart()) {
        MultiVar = true;
        break;
      }
    }
  }

  SourceLocation EndLoc = Lexer::getLocForEndOfToken(
      D->getSourceRange().getEnd(), 0, SM, Result.Context->getLangOpts());
  {
    auto Diag = diag(D->getLocation(), "redundant %0 declaration") << D;
    if (!MultiVar && !DifferentHeaders)
      Diag << FixItHint::CreateRemoval(
          SourceRange(D->getSourceRange().getBegin(), EndLoc));
  }
  diag(Prev->getLocation(), "previously declared here", DiagnosticIDs::Note);
}

} // namespace readability
} // namespace tidy
} // namespace clang
