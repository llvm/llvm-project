//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ShadowedNamespaceFunctionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

template <template <typename> typename ContainerTy>
static auto makeCannonicalTypesRange(const ContainerTy<ParmVarDecl *> &C) {
  return llvm::map_range(C, [](const ParmVarDecl *Param) {
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

namespace {
class ShadowedFunctionFinder : public DynamicRecursiveASTVisitor {
public:
  ShadowedFunctionFinder(const FunctionDecl *GlobalFunc,
                         StringRef GlobalFuncName)
      : GlobalFunc(GlobalFunc), GlobalFuncName(GlobalFuncName) {}

  bool VisitFunctionDecl(FunctionDecl *Func) override {
    // Only process functions that are inside a namespace (not in global scope)
    if (CurrentNamespaceStack.empty())
      return true;

    const NamespaceDecl *CurrentNS = CurrentNamespaceStack.back();

    // TODO: syncronize this check with the matcher?
    if (Func == GlobalFunc || Func->isTemplated() ||
        Func->isThisDeclarationADefinition())
      return true;

    if (Func->getDefinition())
      return true;

    if (Func->getName() == GlobalFuncName && !Func->isVariadic() &&
        hasSameSignature(Func, GlobalFunc)) {
      AllShadowedFuncs.insert(Func);
      if (!ShadowedFunc) {
        ShadowedFunc = Func;
        ShadowedNamespace = CurrentNS;
      }
    }
    return true;
  }

  bool VisitFriendDecl(FriendDecl *Friend) override {
    // Only process functions that are inside a namespace (not in global scope)
    if (CurrentNamespaceStack.empty())
      return true;

    const FunctionDecl *Func =
        dyn_cast_or_null<FunctionDecl>(Friend->getFriendDecl());
    if (!Func) {
      return true;
    }

    const NamespaceDecl *CurrentNS = CurrentNamespaceStack.back();

    // TODO: syncronize this check with the matcher?
    if (Func == GlobalFunc || Func->isTemplated() ||
        Func->isThisDeclarationADefinition())
      return true;

    if (Func->getDefinition())
      return true;

    if (Func->getName() == GlobalFuncName && !Func->isVariadic() &&
        hasSameSignature(Func, GlobalFunc)) {
      // TODO: AllShadowedFuncs
      if (!ShadowedFunc) {
        ShadowedFunc = Func;
        ShadowedNamespace = CurrentNS;
        IsShadowedFuncFriend = true;
      }
    }
    return true;
  }

  bool TraverseNamespaceDecl(NamespaceDecl *NS) override {
    // Skip anonymous namespaces
    if (NS->isAnonymousNamespace())
      return true;

    // Push this namespace onto the stack
    CurrentNamespaceStack.push_back(NS);

    // Traverse children (which will call VisitFunctionDecl for functions
    // inside)
    bool Result = DynamicRecursiveASTVisitor::TraverseNamespaceDecl(NS);

    // Pop the namespace from the stack
    CurrentNamespaceStack.pop_back();

    return Result;
  }

  const FunctionDecl *getShadowedFunc() const { return ShadowedFunc; }
  const NamespaceDecl *getShadowedNamespace() const {
    return ShadowedNamespace;
  }
  const llvm::SmallPtrSet<const FunctionDecl *, 16> &
  getAllShadowedFuncs() const {
    return AllShadowedFuncs;
  }
  bool isShadowedFuncFriend() const { return IsShadowedFuncFriend; }

private:
  const FunctionDecl *GlobalFunc;
  StringRef GlobalFuncName;
  const FunctionDecl *ShadowedFunc = nullptr;
  const NamespaceDecl *ShadowedNamespace = nullptr;
  bool IsShadowedFuncFriend = false;
  llvm::SmallPtrSet<const FunctionDecl *, 16> AllShadowedFuncs;
  llvm::SmallVector<const NamespaceDecl *, 4> CurrentNamespaceStack;
};
} // anonymous namespace

void ShadowedNamespaceFunctionCheck::registerMatchers(MatchFinder *Finder) {
  using ast_matchers::isTemplateInstantiation;
  Finder->addMatcher(functionDecl(isDefinition(),
                                  hasDeclContext(translationUnitDecl()),
                                  unless(anyOf(isImplicit(), isVariadic(),
                                               isMain(), isStaticStorageClass(),
                                               isTemplateInstantiation())))
                         .bind("func"),
                     this);
}

void ShadowedNamespaceFunctionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
  assert(Func);

  const StringRef FuncName = Func->getName();
  if (FuncName.empty())
    return;

  ShadowedFunctionFinder Finder(Func, FuncName);
  Finder.TraverseAST(*Result.Context);

  const FunctionDecl *ShadowedFunc = Finder.getShadowedFunc();
  const NamespaceDecl *ShadowedNamespace = Finder.getShadowedNamespace();
  const auto &AllShadowedFuncs = Finder.getAllShadowedFuncs();
  const bool IsShadowedFuncFriend = Finder.isShadowedFuncFriend();

  if (!ShadowedFunc || !ShadowedNamespace)
    return;

  const bool Ambiguous = AllShadowedFuncs.size() > 1;
  std::string NamespaceName = ShadowedNamespace->getQualifiedNameAsString();
  auto Diag = diag(Func->getLocation(),
                   "free function %0 shadows %select{|at least }1'%2::%3'")
              << Func << Ambiguous << NamespaceName
              << ShadowedFunc->getDeclName().getAsString();

  const SourceLocation NameLoc = Func->getLocation();
  if (NameLoc.isValid() && !Func->getPreviousDecl() && !Ambiguous &&
      !IsShadowedFuncFriend) {
    const std::string Fix = std::move(NamespaceName) + "::";
    Diag << FixItHint::CreateInsertion(NameLoc, Fix);
  }

  for (const FunctionDecl *NoteShadowedFunc : AllShadowedFuncs)
    diag(NoteShadowedFunc->getLocation(), "function %0 declared here",
         DiagnosticIDs::Note)
        << NoteShadowedFunc;
}

} // namespace clang::tidy::misc
