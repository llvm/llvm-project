//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ShadowedNamespaceFunctionCheck.h"
#include "../utils/FixItHintUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tidy;

namespace clang {
namespace tidy {
namespace misc {

void ShadowedNamespaceFunctionCheck::registerMatchers(MatchFinder *Finder) {
  // Simple matcher for all function definitions
  Finder->addMatcher(
      functionDecl(
          isDefinition()
      ).bind("func"),
      this
  );
}

void ShadowedNamespaceFunctionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
  
  if (!Func || !Result.SourceManager)
    return;

  // Skip if not in global namespace
  const DeclContext *DC = Func->getDeclContext();
  if (!DC->isTranslationUnit())
    return;

  // Skip templates, static functions, main, etc.
  if (Func->isTemplated() || Func->isStatic() || 
      Func->isMain() || Func->isImplicit())
    return;

  const std::string FuncName = Func->getNameAsString();
  if (FuncName.empty())
    return;

  const ASTContext *Context = Result.Context;

  // Look for functions with the same name in namespaces
  const FunctionDecl *ShadowedFunc = nullptr;
  const NamespaceDecl *ShadowedNamespace = nullptr;

  // Traverse all declarations in the translation unit
  for (const auto *Decl : Context->getTranslationUnitDecl()->decls()) {
    if (const auto *NS = dyn_cast<NamespaceDecl>(Decl)) {
      std::tie(ShadowedFunc, ShadowedNamespace) = findShadowedInNamespace(NS, Func, FuncName);
      if (ShadowedFunc) break;
    }
  }

  if (!ShadowedFunc || !ShadowedNamespace)
    return;

  if (ShadowedFunc->getDefinition())
    return;

  // Generate warning message
  const std::string NamespaceName = ShadowedNamespace->getQualifiedNameAsString();
  auto Diag = diag(Func->getLocation(), 
                   "free function %0 shadows '%1::%2'")
              << Func->getDeclName() 
              << NamespaceName 
              << ShadowedFunc->getDeclName().getAsString();

  // Generate fixit hint to add namespace qualification
  const SourceLocation NameLoc = Func->getLocation();
  if (NameLoc.isValid() && !Func->getPreviousDecl()) {
    const std::string Fix = NamespaceName + "::";
    Diag << FixItHint::CreateInsertion(NameLoc, Fix);
  }

  // Note: Also show where the shadowed function is declared
  diag(ShadowedFunc->getLocation(), 
       "function %0 declared here", DiagnosticIDs::Note)
      << ShadowedFunc->getDeclName();
}

std::pair<const FunctionDecl *, const NamespaceDecl *> ShadowedNamespaceFunctionCheck::findShadowedInNamespace(
    const NamespaceDecl *NS, 
    const FunctionDecl *GlobalFunc,
    const std::string &GlobalFuncName) {
  
  // Skip anonymous namespaces
  if (NS->isAnonymousNamespace())
    return {nullptr, nullptr};

  for (const auto *Decl : NS->decls()) {
    // Check nested namespaces
    if (const auto *NestedNS = dyn_cast<NamespaceDecl>(Decl)) {
      auto [ShadowedFunc, ShadowedNamespace] = findShadowedInNamespace(NestedNS, GlobalFunc, GlobalFuncName);
      if (ShadowedFunc)
        return {ShadowedFunc, ShadowedNamespace};
    }

    // Check functions
    if (const auto *Func = dyn_cast<FunctionDecl>(Decl)) {
      // Skip if it's the same function, templates, or definitions
      if (Func == GlobalFunc || Func->isTemplated() || 
          Func->isThisDeclarationADefinition())
        continue;

      if (Func->getNameAsString() == GlobalFuncName) {
        return {Func, NS};
      }
    }
  }
  return {nullptr, nullptr};
}

} // namespace misc
} // namespace tidy
} // namespace clang

