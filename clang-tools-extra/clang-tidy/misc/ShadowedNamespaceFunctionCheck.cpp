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

template <typename R> static auto makeCannonicalTypesRange(R &&r) {
  return llvm::map_range(r, [](const ParmVarDecl *Param) {
    return Param->getType().getCanonicalType();
  });
}

static bool hasSameSignature(const FunctionDecl *Func1,
                             const FunctionDecl *Func2) {
  if (Func1->param_size() != Func2->param_size())
    return false;

  if (Func1->getReturnType().getCanonicalType() !=
      Func2->getReturnType().getCanonicalType())
    return false;

  return llvm::equal(makeCannonicalTypesRange(Func1->parameters()),
                     makeCannonicalTypesRange(Func2->parameters()));
}

static std::pair<const FunctionDecl *, const NamespaceDecl *>
findShadowedInNamespace(const NamespaceDecl *NS, const FunctionDecl *GlobalFunc,
                        StringRef GlobalFuncName,
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

      if (Func->getName() == GlobalFuncName && !Func->isVariadic() &&
          hasSameSignature(Func, GlobalFunc)) {
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

  const StringRef FuncName = Func->getName();
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
  std::string NamespaceName = ShadowedNamespace->getQualifiedNameAsString();
  auto Diag = diag(Func->getLocation(),
                   "free function %0 shadows %select{|at least }1'%2::%3'")
              << Func << Ambiguous << NamespaceName
              << ShadowedFunc->getDeclName().getAsString();

  const SourceLocation NameLoc = Func->getLocation();
  if (NameLoc.isValid() && !Func->getPreviousDecl() && !Ambiguous) {
    const std::string Fix = std::move(NamespaceName) + "::";
    Diag << FixItHint::CreateInsertion(NameLoc, Fix);
  }

  for (const FunctionDecl *NoteShadowedFunc : AllShadowedFuncs)
    diag(NoteShadowedFunc->getLocation(), "function %0 declared here",
         DiagnosticIDs::Note)
        << NoteShadowedFunc->getDeclName();
}

} // namespace clang::tidy::misc
