//===--- RedundantDeclarationCheck.cpp - clang-tidy------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

AST_MATCHER(FunctionDecl, doesDeclarationForceExternallyVisibleDefinition) {
  return Node.doesDeclarationForceExternallyVisibleDefinition();
}

RedundantDeclarationCheck::RedundantDeclarationCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", true)) {}

void RedundantDeclarationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void RedundantDeclarationCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      namedDecl(anyOf(varDecl(unless(isDefinition())),
                      functionDecl(unless(anyOf(
                          isDefinition(), isDefaulted(),
                          doesDeclarationForceExternallyVisibleDefinition(),
                          hasParent(friendDecl()))))))
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
  if (IgnoreMacros &&
      (D->getLocation().isMacroID() || Prev->getLocation().isMacroID()))
    return;
  // Don't complain when the previous declaration is a friend declaration.
  for (const auto &Parent : Result.Context->getParents(*Prev))
    if (Parent.get<FriendDecl>())
      return;

  const SourceManager &SM = *Result.SourceManager;

  const bool DifferentHeaders =
      !SM.isInMainFile(D->getLocation()) &&
      !SM.isWrittenInSameFile(Prev->getLocation(), D->getLocation());

  bool MultiVar = false;
  if (const auto *VD = dyn_cast<VarDecl>(D)) {
    // Is this a multivariable declaration?
    for (const auto Other : VD->getDeclContext()->decls()) {
      if (Other != D && Other->getBeginLoc() == VD->getBeginLoc()) {
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
