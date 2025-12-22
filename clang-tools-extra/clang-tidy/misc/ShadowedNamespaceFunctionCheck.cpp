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

static bool sholdBeIgnored(const FunctionDecl *Func, bool IgnoreTemplated) {
  if (Func->isVariadic())
    return true;
  if (IgnoreTemplated && Func->isTemplated())
    return true;
  return false;
}

namespace {
class ShadowedFunctionFinder : public DynamicRecursiveASTVisitor {
public:
  ShadowedFunctionFinder(const FunctionDecl *GlobalFunc,
                         StringRef GlobalFuncName, bool IgnoreTemplated)
      : GlobalFunc(GlobalFunc), GlobalFuncName(GlobalFuncName),
        IgnoreTemplated(IgnoreTemplated) {}

  bool VisitFunctionDecl(FunctionDecl *Func) override {
    // Only process functions that are inside a namespace (not in global scope)
    if (CurrentNamespaceStack.empty())
      return true;

    if (isa<CXXMethodDecl>(Func))
      return true;

    if (Func->getDefinition() || sholdBeIgnored(Func, IgnoreTemplated))
      return true;

    const NamespaceDecl *CurrentNS = CurrentNamespaceStack.back();

    if (Func != GlobalFunc && Func->getNameAsString() == GlobalFuncName &&
        hasSameSignature(Func, GlobalFunc)) {
      AllShadowedFuncs.insert(Func);
      if (!ShadowedFunc) {
        if (Func->isInIdentifierNamespace(Decl::IDNS_OrdinaryFriend) ||
            Func->isInIdentifierNamespace(Decl::IDNS_TagFriend))
          IsShadowedFuncFriend = true;
        ShadowedFunc = Func;
        ShadowedNamespace = CurrentNS;
      }
    }
    return true;
  }

  bool TraverseNamespaceDecl(NamespaceDecl *NS) override {
    if (NS->isAnonymousNamespace())
      return true;

    CurrentNamespaceStack.push_back(NS);

    // Traverse children (which will call VisitFunctionDecl for functions
    // inside)
    const bool Result = DynamicRecursiveASTVisitor::TraverseNamespaceDecl(NS);

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
  const FunctionDecl *const GlobalFunc;
  const StringRef GlobalFuncName;
  const bool IgnoreTemplated;
  const FunctionDecl *ShadowedFunc = nullptr;
  const NamespaceDecl *ShadowedNamespace = nullptr;
  bool IsShadowedFuncFriend = false;
  llvm::SmallPtrSet<const FunctionDecl *, 16> AllShadowedFuncs;
  llvm::SmallVector<const NamespaceDecl *, 4> CurrentNamespaceStack;
};
} // anonymous namespace

void ShadowedNamespaceFunctionCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(
          isDefinition(), hasDeclContext(translationUnitDecl()),
          unless(anyOf(isImplicit(), isMain(), isStaticStorageClass())))
          .bind("func"),
      this);
}

void ShadowedNamespaceFunctionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
  assert(Func);

  const std::string FuncName = Func->getNameAsString();
  if (FuncName.empty() || sholdBeIgnored(Func, IgnoreTemplated))
    return;

  ShadowedFunctionFinder Finder(Func, FuncName, IgnoreTemplated);
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
