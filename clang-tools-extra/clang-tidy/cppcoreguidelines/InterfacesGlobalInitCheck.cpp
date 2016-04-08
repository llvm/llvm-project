//===--- InterfacesGlobalInitCheck.cpp - clang-tidy------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InterfacesGlobalInitCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

void InterfacesGlobalInitCheck::registerMatchers(MatchFinder *Finder) {
  const auto IsGlobal =
      allOf(hasGlobalStorage(),
            hasDeclContext(anyOf(translationUnitDecl(), // Global scope.
                                 namespaceDecl(),       // Namespace scope.
                                 recordDecl())),        // Class scope.
            unless(isConstexpr()));

  const auto ReferencesUndefinedGlobalVar = declRefExpr(hasDeclaration(
      varDecl(IsGlobal, unless(isDefinition())).bind("referencee")));

  Finder->addMatcher(
      varDecl(IsGlobal, isDefinition(),
              hasInitializer(expr(hasDescendant(ReferencesUndefinedGlobalVar))))
          .bind("var"),
      this);
}

void InterfacesGlobalInitCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *const Var = Result.Nodes.getNodeAs<VarDecl>("var");
  // For now assume that people who write macros know what they're doing.
  if (Var->getLocation().isMacroID())
    return;
  const auto *const Referencee = Result.Nodes.getNodeAs<VarDecl>("referencee");
  // If the variable has been defined, we're good.
  const auto *const ReferenceeDef = Referencee->getDefinition();
  if (ReferenceeDef != nullptr &&
      Result.SourceManager->isBeforeInTranslationUnit(
          ReferenceeDef->getLocation(), Var->getLocation())) {
    return;
  }
  diag(Var->getLocation(),
       "initializing non-local variable with non-const expression depending on "
       "uninitialized non-local variable %0")
      << Referencee;
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
