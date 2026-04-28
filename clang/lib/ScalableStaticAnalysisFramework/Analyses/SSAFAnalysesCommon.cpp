//===- SSAFAnalysesCommon.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSAFAnalysesCommon.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include <set>

using namespace clang;

namespace {
// Traverses the AST and finds contributors.
class ContributorFinder : public DynamicRecursiveASTVisitor {
public:
  std::set<const NamedDecl *> Contributors;

  bool VisitFunctionDecl(FunctionDecl *D) override {
    Contributors.insert(D);
    return true;
  }

  bool VisitRecordDecl(RecordDecl *D) override {
    Contributors.insert(D);
    return true;
  }

  bool VisitVarDecl(VarDecl *D) override {
    DeclContext *DC = D->getDeclContext();

    if (DC->isFileContext() || DC->isNamespace())
      Contributors.insert(D);
    return true;
  }
};

/// An AST visitor that skips the root node's strict-descendants that are
/// callable Decls and record Decls, because those are separate contributors.
///
/// Clients need to implement their own "MatchAction", which is a function that
/// takes a `DynTypedNode`, decides if the node matches and performs any further
/// callback actions.
/// ContributorFactFinder takes a reference to a "MatchAction". It does not own
/// the "MatchAction", which is usually stateful and may own containers.
class ContributorFactFinder : public DynamicRecursiveASTVisitor {
  llvm::function_ref<void(const DynTypedNode &)> MatchActionRef;
  const NamedDecl *RootDecl = nullptr;

  template <typename NodeTy> void match(const NodeTy &Node) {
    MatchActionRef(DynTypedNode::create(Node));
  }

public:
  ContributorFactFinder(
      llvm::function_ref<void(const DynTypedNode &)> MatchActionRef)
      : MatchActionRef(MatchActionRef) {
    ShouldVisitTemplateInstantiations = true;
    ShouldVisitImplicitCode = false;
  }

  // The entry point:
  void findMatches(const NamedDecl *Contributor) {
    RootDecl = Contributor;
    TraverseDecl(const_cast<NamedDecl *>(Contributor));
  }

  bool TraverseDecl(Decl *Node) override {
    if (!Node)
      return true;
    // To skip callables:
    if (Node != RootDecl &&
        isa<FunctionDecl, BlockDecl, ObjCMethodDecl, RecordDecl>(Node))
      return true;
    match(*Node);
    return DynamicRecursiveASTVisitor::TraverseDecl(Node);
  }

  bool TraverseStmt(Stmt *Node) override {
    if (!Node)
      return true;
    match(*Node);
    return DynamicRecursiveASTVisitor::TraverseStmt(Node);
  }

  bool TraverseLambdaExpr(LambdaExpr *L) override {
    // TODO: lambda captures of pointer variables (by copy or by reference)
    // are currently not tracked. Each capture initializes an implicit closure
    // field from the captured variable, which constitutes a pointer assignment
    // edge that should be recorded here.
    return true; // Skip lambda as it is a callable.
  }
};
} // namespace

void clang::ssaf::findContributors(
    ASTContext &Ctx, std::vector<const NamedDecl *> &Contributors) {
  ContributorFinder Finder;
  Finder.TraverseAST(Ctx);
  Contributors.insert(Contributors.end(), Finder.Contributors.begin(),
                      Finder.Contributors.end());
}

void clang::ssaf::findMatchesIn(
    const NamedDecl *Contributor,
    llvm::function_ref<void(const DynTypedNode &)> MatchActionRef) {
  ContributorFactFinder{MatchActionRef}.findMatches(Contributor);
}

llvm::Error clang::ssaf::makeEntityNameErr(clang::ASTContext &Ctx,
                                           const clang::NamedDecl *D) {
  return makeErrAtNode(Ctx, D, "failed to create entity name for %s",
                       D->getNameAsString().data());
}
