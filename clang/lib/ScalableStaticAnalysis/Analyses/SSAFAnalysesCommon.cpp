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
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/SSAFOptions.h"
#include "llvm/ADT/SetVector.h"

using namespace clang;
using namespace ssaf;

std::string ssaf::describeJSONValue(const llvm::json::Value &V) {
  return llvm::formatv("{0:2}", V).str();
}

std::string ssaf::describeJSONValue(const llvm::json::Array &A) {
  return llvm::formatv("array of size {0}", A.size()).str();
}

std::string ssaf::describeJSONValue(const llvm::json::Object &O) {
  return llvm::formatv("an object of {0} key(s)", O.size()).str();
}

namespace {
// Traverses the AST and finds contributors.
class ContributorFinder : public DynamicRecursiveASTVisitor {
public:
  llvm::SetVector<const NamedDecl *> Contributors;
  const SSAFOptions &Opts;

  ContributorFinder(ASTContext &Ctx, const SSAFOptions &Opts,
                    bool ExtractFromSystemHeaders)
      : Opts(Opts), Ctx(Ctx),
        ExtractFromSystemHeaders(ExtractFromSystemHeaders) {
    ShouldVisitTemplateInstantiations = true;
    ShouldVisitImplicitCode = false;
  }

  bool VisitFunctionDecl(FunctionDecl *D) override {
    if (!skipForSystemHeader(D))
      Contributors.insert(D);
    return true;
  }

  bool VisitRecordDecl(RecordDecl *D) override {
    if (skipForSystemHeader(D))
      return true;
    Contributors.insert(D);
    return true;
  }

  bool VisitVarDecl(VarDecl *D) override {
    if (skipForSystemHeader(D))
      return true;
    DeclContext *DC = D->getDeclContext();

    // Collects Decl for global variables or static data members:
    if (DC->isFileContext() || D->isStaticDataMember()) {
      Contributors.insert(D);
      return true;
    }

    // Optionally include block-scope (function-local) variables. Parameters
    // are intentionally skipped: they are exposed via their parent function's
    // USR + a parameter-index suffix in getEntityName, so registering them as
    // independent contributors would be redundant.
    //
    // FIXME: clang::index::generateUSRForDecl can produce non-unique or empty
    // USRs for some local declaration shapes (e.g., locals of certain template
    // instantiations). The current addEntity path returns std::nullopt when
    // that happens and downstream extractors skip gracefully, so this is
    // tolerated for now.
    if (Opts.IncludeLocalEntities && !D->isImplicit() && !isa<ParmVarDecl>(D) &&
        DC->isFunctionOrMethod())
      Contributors.insert(D);
    return true;
  }

  bool VisitLambdaExpr(LambdaExpr *L) override {
    return VisitFunctionDecl(L->getCallOperator());
  }

private:
  bool skipForSystemHeader(const Decl *D) const {
    if (ExtractFromSystemHeaders)
      return false;
    SourceLocation Loc = D->getLocation();
    return Loc.isValid() && Ctx.getSourceManager().isInSystemHeader(Loc);
  }

  ASTContext &Ctx;
  bool ExtractFromSystemHeaders;
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

void ssaf::findContributors(
    ASTContext &Ctx, const SSAFOptions &Options,
    llvm::DenseMap<const NamedDecl *, std::vector<const NamedDecl *>>
        &Contributors,
    bool ExtractFromSystemHeaders) {
  ContributorFinder Finder{Ctx, Options, ExtractFromSystemHeaders};
  Finder.TraverseAST(Ctx);
  for (const NamedDecl *C : Finder.Contributors)
    Contributors[cast<NamedDecl>(C->getCanonicalDecl())].push_back(C);
}

void ssaf::findMatchesIn(
    const NamedDecl *Contributor,
    llvm::function_ref<void(const DynTypedNode &)> MatchActionRef) {
  ContributorFactFinder{MatchActionRef}.findMatches(Contributor);
}

llvm::Error clang::ssaf::makeEntityNameErr(clang::ASTContext &Ctx,
                                           const clang::NamedDecl *D) {
  return makeErrAtNode(Ctx, D, "failed to create entity name for %s",
                       D->getNameAsString().data());
}
