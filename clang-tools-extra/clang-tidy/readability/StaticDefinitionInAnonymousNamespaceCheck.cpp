//===--- StaticDefinitionInAnonymousNamespaceCheck.cpp - clang-tidy--------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "StaticDefinitionInAnonymousNamespaceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {
AST_POLYMORPHIC_MATCHER(isStatic, AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                                  VarDecl)) {
  return Node.getStorageClass() == SC_Static;
}
} // namespace

void StaticDefinitionInAnonymousNamespaceCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(namedDecl(anyOf(functionDecl(isDefinition(), isStatic()),
                                     varDecl(isDefinition(), isStatic())),
                               hasParent(namespaceDecl(isAnonymous())))
                         .bind("static-def"),
                     this);
}

void StaticDefinitionInAnonymousNamespaceCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Def = Result.Nodes.getNodeAs<NamedDecl>("static-def");
  // Skips all static definitions defined in Macro.
  if (Def->getLocation().isMacroID())
    return;

  // Skips all static definitions in function scope.
  const DeclContext *DC = Def->getDeclContext();
  if (DC->getDeclKind() != Decl::Namespace)
    return;

  auto Diag =
      diag(Def->getLocation(), "%0 is a static definition in "
                               "anonymous namespace; static is redundant here")
      << Def;
  Token Tok;
  SourceLocation Loc = Def->getSourceRange().getBegin();
  while (Loc < Def->getSourceRange().getEnd() &&
         !Lexer::getRawToken(Loc, Tok, *Result.SourceManager, getLangOpts(),
                             true)) {
    SourceRange TokenRange(Tok.getLocation(), Tok.getEndLoc());
    StringRef SourceText =
        Lexer::getSourceText(CharSourceRange::getTokenRange(TokenRange),
                             *Result.SourceManager, getLangOpts());
    if (SourceText == "static") {
      Diag << FixItHint::CreateRemoval(TokenRange);
      break;
    }
    Loc = Tok.getEndLoc();
  }
}

} // namespace readability
} // namespace tidy
} // namespace clang
