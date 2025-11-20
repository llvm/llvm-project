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
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tidy;

namespace clang::tidy::misc {

static bool hasSameParameters(const FunctionDecl *Func1,
                              const FunctionDecl *Func2) {
  if (Func1->param_size() != Func2->param_size())
    return false;

  return llvm::all_of_zip(
      Func1->parameters(), Func2->parameters(),
      [](const ParmVarDecl *Param1, const ParmVarDecl *Param2) {
        return Param1->getType().getCanonicalType() ==
               Param2->getType().getCanonicalType();
      });
}

static std::pair<const FunctionDecl *, const NamespaceDecl *>
findShadowedInNamespace(const NamespaceDecl *NS, const FunctionDecl *GlobalFunc,
                        const std::string &GlobalFuncName,
                        llvm::SmallPtrSet<const FunctionDecl *, 16> &All) {

  if (NS->isAnonymousNamespace())
    return {nullptr, nullptr};

  const FunctionDecl *ShadowedFunc = nullptr;
  const NamespaceDecl *ShadowedNamespace = nullptr;

  for (const auto *Decl : NS->decls()) {
    // Check nested namespaces
    if (const auto *NestedNS = dyn_cast<NamespaceDecl>(Decl)) {
      auto [NestedShadowedFunc, NestedShadowedNamespace] =
          findShadowedInNamespace(NestedNS, GlobalFunc, GlobalFuncName, All);
      if (!ShadowedFunc)
        std::tie(ShadowedFunc, ShadowedNamespace) =
            std::tie(NestedShadowedFunc, NestedShadowedNamespace);
    }

    // Check functions
    if (const auto *Func = dyn_cast<FunctionDecl>(Decl)) {
      // TODO: syncronize this check with the matcher?
      if (Func == GlobalFunc || Func->isTemplated() ||
          Func->isThisDeclarationADefinition())
        continue;

      if (Func->getNameAsString() == GlobalFuncName && !Func->isVariadic() &&
          hasSameParameters(Func, GlobalFunc) &&
          Func->getReturnType().getCanonicalType() ==
              GlobalFunc->getReturnType().getCanonicalType()) {
        All.insert(Func);
        if (!ShadowedFunc)
          std::tie(ShadowedFunc, ShadowedNamespace) = std::tie(Func, NS);
      }
    }
  }
  return {ShadowedFunc, ShadowedNamespace};
}

void ShadowedNamespaceFunctionCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(isDefinition(), decl(hasDeclContext(translationUnitDecl())),
                   unless(anyOf(isImplicit(), isVariadic(), isMain(),
                                isStaticStorageClass(),
                                ast_matchers::isTemplateInstantiation())))
          .bind("func"),
      this);
}

void ShadowedNamespaceFunctionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");

  const std::string FuncName = Func->getNameAsString();
  if (FuncName.empty())
    return;

  const ASTContext *Context = Result.Context;

  llvm::SmallPtrSet<const FunctionDecl *, 16> AllShadowedFuncs;
  const FunctionDecl *ShadowedFunc = nullptr;
  const NamespaceDecl *ShadowedNamespace = nullptr;

  for (const auto *Decl : Context->getTranslationUnitDecl()->decls()) {
    if (const auto *NS = dyn_cast<NamespaceDecl>(Decl)) {
      auto [NestedShadowedFunc, NestedShadowedNamespace] =
          findShadowedInNamespace(NS, Func, FuncName, AllShadowedFuncs);
      if (!ShadowedFunc)
        std::tie(ShadowedFunc, ShadowedNamespace) =
            std::tie(NestedShadowedFunc, NestedShadowedNamespace);
    }
  }

  if (!ShadowedFunc || !ShadowedNamespace)
    return;

  // TODO: should it be inside findShadowedInNamespace?
  if (ShadowedFunc->getDefinition())
    return;

  const bool Ambiguous = AllShadowedFuncs.size() > 1;
  const std::string NamespaceName =
      ShadowedNamespace->getQualifiedNameAsString();
  auto Diag = diag(Func->getLocation(),
                   "free function %0 shadows %select{|at least }1'%2::%3'")
              << Func->getDeclName() << Ambiguous << NamespaceName
              << ShadowedFunc->getDeclName().getAsString();

  const SourceLocation NameLoc = Func->getLocation();
  if (NameLoc.isValid() && !Func->getPreviousDecl() && !Ambiguous) {
    const std::string Fix = NamespaceName + "::";
    Diag << FixItHint::CreateInsertion(NameLoc, Fix);
  }

  for (const FunctionDecl *NoteShadowedFunc : AllShadowedFuncs)
    diag(NoteShadowedFunc->getLocation(), "function %0 declared here",
         DiagnosticIDs::Note)
        << NoteShadowedFunc->getDeclName();
}

} // namespace clang::tidy::misc
