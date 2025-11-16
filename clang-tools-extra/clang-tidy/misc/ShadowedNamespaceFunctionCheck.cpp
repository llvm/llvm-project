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
      Func->getName() == "main" || Func->isImplicit())
    return;

  std::string FuncName = Func->getNameAsString();
  if (FuncName.empty())
    return;

  ASTContext *Context = Result.Context;

  // Look for functions with the same name in namespaces
  const FunctionDecl *ShadowedFunc = nullptr;
  const NamespaceDecl *ShadowedNamespace = nullptr;

  // Traverse all declarations in the translation unit
  for (const auto *Decl : Context->getTranslationUnitDecl()->decls()) {
    if (const auto *NS = dyn_cast<NamespaceDecl>(Decl)) {
      findShadowedInNamespace(NS, Func, FuncName, ShadowedFunc, ShadowedNamespace);
      if (ShadowedFunc) break;
    }
  }

  if (!ShadowedFunc || !ShadowedNamespace)
    return;

  // Generate warning message
  std::string NamespaceName = ShadowedNamespace->getQualifiedNameAsString();
  auto Diag = diag(Func->getLocation(), 
                   "free function %0 shadows '%1::%2'")
              << Func->getDeclName() 
              << NamespaceName 
              << ShadowedFunc->getDeclName().getAsString();

  // Generate fixit hint to add namespace qualification
  SourceLocation NameLoc = Func->getLocation();
  if (NameLoc.isValid() && !Func->getPreviousDecl()) {
    std::string Fix = NamespaceName + "::";
    Diag << FixItHint::CreateInsertion(NameLoc, Fix);
  }

  // Note: Also show where the shadowed function is declared
  diag(ShadowedFunc->getLocation(), 
       "function %0 declared here", DiagnosticIDs::Note)
      << ShadowedFunc->getDeclName();
}

void ShadowedNamespaceFunctionCheck::findShadowedInNamespace(
    const NamespaceDecl *NS, 
    const FunctionDecl *GlobalFunc,
    const std::string &GlobalFuncName,
    const FunctionDecl *&ShadowedFunc,
    const NamespaceDecl *&ShadowedNamespace) {
  
  // Skip anonymous namespaces
  if (NS->isAnonymousNamespace())
    return;

  for (const auto *Decl : NS->decls()) {
    // Check nested namespaces
    if (const auto *NestedNS = dyn_cast<NamespaceDecl>(Decl)) {
      findShadowedInNamespace(NestedNS, GlobalFunc, GlobalFuncName, 
                             ShadowedFunc, ShadowedNamespace);
      if (ShadowedFunc) return;
    }

    // Check functions
    if (const auto *Func = dyn_cast<FunctionDecl>(Decl)) {
      // Skip if it's the same function, templates, or definitions
      if (Func == GlobalFunc || Func->isTemplated() || 
          Func->isThisDeclarationADefinition())
        continue;

      if (Func->getNameAsString() == GlobalFuncName) {
        ShadowedFunc = Func;
        ShadowedNamespace = NS;
        return;
      }
    }
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang

