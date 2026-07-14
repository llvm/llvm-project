//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UsePlaceholderBindingCheck.h"
#include "../utils/DeclRefExprUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {
AST_POLYMORPHIC_MATCHER(isInMacro,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(Stmt, Decl)) {
  return Node.getBeginLoc().isMacroID() || Node.getEndLoc().isMacroID();
}

AST_MATCHER(VarDecl, hasAutomaticStorageDurationAndNoSpecifiers) {
  return !Node.hasAttrs() && Node.getStorageClass() == SC_None &&
         Node.getTSCSpec() == TSCS_unspecified;
}
} // namespace

void UsePlaceholderBindingCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cStyleCastExpr(
          unless(isInMacro()), hasType(voidType()),
          hasParent(stmt(anyOf(compoundStmt(), switchCase()))),
          hasSourceExpression(ignoringParens(declRefExpr(
              unless(isInMacro()),
              to(bindingDecl(forDecomposition(varDecl(
                                 unless(isInMacro()),
                                 hasAutomaticStorageDurationAndNoSpecifiers(),
                                 hasAncestor(compoundStmt().bind("scope")))))
                     .bind("binding"))))))
          .bind("cast"),
      this);
}

void UsePlaceholderBindingCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Cast = Result.Nodes.getNodeAs<CStyleCastExpr>("cast");
  const auto *Binding = Result.Nodes.getNodeAs<BindingDecl>("binding");
  const auto *Scope = Result.Nodes.getNodeAs<CompoundStmt>("scope");
  ASTContext &Context = *Result.Context;

  if (Binding->isParameterPack() ||
      Binding->isPlaceholderVar(Context.getLangOpts()))
    return;

  const llvm::SmallPtrSet<const DeclRefExpr *, 16> Refs =
      utils::decl_ref_expr::allDeclRefExprs(*Binding, *Scope, Context);
  if (Refs.size() != 1)
    return;

  const SourceLocation SemiLoc = Lexer::findLocationAfterToken(
      Cast->getEndLoc(), tok::semi, Context.getSourceManager(),
      Context.getLangOpts(), /*SkipTrailingWhitespaceAndNewLine=*/true);
  if (SemiLoc.isInvalid())
    return;

  diag(Binding->getLocation(),
       "binding %0 is only used to suppress an unused variable warning; "
       "use a placeholder '_' instead")
      << Binding << FixItHint::CreateReplacement(Binding->getSourceRange(), "_")
      << FixItHint::CreateRemoval(
             CharSourceRange::getCharRange(Cast->getBeginLoc(), SemiLoc));
}

} // namespace clang::tidy::modernize
