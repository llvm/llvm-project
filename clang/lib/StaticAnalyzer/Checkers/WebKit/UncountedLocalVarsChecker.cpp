//=======- UncountedLocalVarsChecker.cpp -------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTUtils.h"
#include "DiagOutputUtils.h"
#include "PtrTypesSemantics.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include <optional>

using namespace clang;
using namespace ento;

namespace {

// FIXME: should be defined by anotations in the future
bool isRefcountedStringsHack(const VarDecl *V) {
  assert(V);
  auto safeClass = [](const std::string &className) {
    return className == "String" || className == "AtomString" ||
           className == "UniquedString" || className == "Identifier";
  };
  QualType QT = V->getType();
  auto *T = QT.getTypePtr();
  if (auto *CXXRD = T->getAsCXXRecordDecl()) {
    if (safeClass(safeGetName(CXXRD)))
      return true;
  }
  if (T->isPointerType() || T->isReferenceType()) {
    if (auto *CXXRD = T->getPointeeCXXRecordDecl()) {
      if (safeClass(safeGetName(CXXRD)))
        return true;
    }
  }
  return false;
}

bool isGuardedScopeEmbeddedInGuardianScope(const VarDecl *Guarded,
                                           const VarDecl *MaybeGuardian) {
  assert(Guarded);
  assert(MaybeGuardian);

  if (!MaybeGuardian->isLocalVarDecl())
    return false;

  const CompoundStmt *guardiansClosestCompStmtAncestor = nullptr;

  ASTContext &ctx = MaybeGuardian->getASTContext();

  for (DynTypedNodeList guardianAncestors = ctx.getParents(*MaybeGuardian);
       !guardianAncestors.empty();
       guardianAncestors = ctx.getParents(
           *guardianAncestors
                .begin()) // FIXME - should we handle all of the parents?
  ) {
    for (auto &guardianAncestor : guardianAncestors) {
      if (auto *CStmtParentAncestor = guardianAncestor.get<CompoundStmt>()) {
        guardiansClosestCompStmtAncestor = CStmtParentAncestor;
        break;
      }
    }
    if (guardiansClosestCompStmtAncestor)
      break;
  }

  if (!guardiansClosestCompStmtAncestor)
    return false;

  // We need to skip the first CompoundStmt to avoid situation when guardian is
  // defined in the same scope as guarded variable.
  bool HaveSkippedFirstCompoundStmt = false;
  for (DynTypedNodeList guardedVarAncestors = ctx.getParents(*Guarded);
       !guardedVarAncestors.empty();
       guardedVarAncestors = ctx.getParents(
           *guardedVarAncestors
                .begin()) // FIXME - should we handle all of the parents?
  ) {
    for (auto &guardedVarAncestor : guardedVarAncestors) {
      if (auto *CStmtAncestor = guardedVarAncestor.get<CompoundStmt>()) {
        if (!HaveSkippedFirstCompoundStmt) {
          HaveSkippedFirstCompoundStmt = true;
          continue;
        }
        if (CStmtAncestor == guardiansClosestCompStmtAncestor)
          return true;
      }
    }
  }

  return false;
}

class UncountedLocalVarsChecker
    : public Checker<check::ASTDecl<TranslationUnitDecl>> {
  BugType Bug{this,
              "Uncounted raw pointer or reference not provably backed by "
              "ref-counted variable",
              "WebKit coding guidelines"};
  mutable BugReporter *BR;

  TrivialFunctionAnalysis TFA;

public:
  void checkASTDecl(const TranslationUnitDecl *TUD, AnalysisManager &MGR,
                    BugReporter &BRArg) const {
    BR = &BRArg;

    // The calls to checkAST* from AnalysisConsumer don't
    // visit template instantiations or lambda classes. We
    // want to visit those, so we make our own RecursiveASTVisitor.
    struct LocalVisitor : public RecursiveASTVisitor<LocalVisitor> {
      const UncountedLocalVarsChecker *Checker;
      explicit LocalVisitor(const UncountedLocalVarsChecker *Checker)
          : Checker(Checker) {
        assert(Checker);
      }

      bool shouldVisitTemplateInstantiations() const { return true; }
      bool shouldVisitImplicitCode() const { return false; }

      bool VisitVarDecl(VarDecl *V) {
        Checker->visitVarDecl(V);
        return true;
      }
    };

    LocalVisitor visitor(this);
    visitor.TraverseDecl(const_cast<TranslationUnitDecl *>(TUD));
  }

  void visitVarDecl(const VarDecl *V) const {
    if (shouldSkipVarDecl(V))
      return;

    const auto *ArgType = V->getType().getTypePtr();
    if (!ArgType)
      return;

    std::optional<bool> IsUncountedPtr = isUncountedPtr(ArgType);
    if (IsUncountedPtr && *IsUncountedPtr) {

      ASTContext &ctx = V->getASTContext();
      for (DynTypedNodeList ancestors = ctx.getParents(*V); !ancestors.empty();
           ancestors = ctx.getParents(*ancestors.begin())) {
        for (auto &ancestor : ancestors) {
          if (auto *S = ancestor.get<IfStmt>(); S && TFA.isTrivial(S))
            return;
          if (auto *S = ancestor.get<ForStmt>(); S && TFA.isTrivial(S))
            return;
          if (auto *S = ancestor.get<CXXForRangeStmt>(); S && TFA.isTrivial(S))
            return;
          if (auto *S = ancestor.get<WhileStmt>(); S && TFA.isTrivial(S))
            return;
          if (auto *S = ancestor.get<CompoundStmt>(); S && TFA.isTrivial(S))
            return;
        }
      }

      const Expr *const InitExpr = V->getInit();
      if (!InitExpr)
        return; // FIXME: later on we might warn on uninitialized vars too

      const clang::Expr *const InitArgOrigin =
          tryToFindPtrOrigin(InitExpr, /*StopAtFirstRefCountedObj=*/false)
              .first;

      if (!InitArgOrigin)
        return;

      if (isa<CXXThisExpr>(InitArgOrigin))
        return;

      if (auto *Ref = llvm::dyn_cast<DeclRefExpr>(InitArgOrigin)) {
        if (auto *MaybeGuardian =
                dyn_cast_or_null<VarDecl>(Ref->getFoundDecl())) {
          const auto *MaybeGuardianArgType =
              MaybeGuardian->getType().getTypePtr();
          if (MaybeGuardianArgType) {
            const CXXRecordDecl *const MaybeGuardianArgCXXRecord =
                MaybeGuardianArgType->getAsCXXRecordDecl();
            if (MaybeGuardianArgCXXRecord) {
              if (MaybeGuardian->isLocalVarDecl() &&
                  (isRefCounted(MaybeGuardianArgCXXRecord) ||
                   isRefcountedStringsHack(MaybeGuardian)) &&
                  isGuardedScopeEmbeddedInGuardianScope(V, MaybeGuardian))
                return;
            }
          }
          // Parameters are guaranteed to be safe for the duration of the call
          // by another checker.
          if (isa<ParmVarDecl>(MaybeGuardian))
            return;
        }
      }

      reportBug(V);
    }
  }

  bool shouldSkipVarDecl(const VarDecl *V) const {
    assert(V);
    if (!V->isLocalVarDecl())
      return true;

    return false;
  }

  void reportBug(const VarDecl *V) const {
    assert(V);
    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    Os << "Local variable ";
    printQuotedQualifiedName(Os, V);
    Os << " is uncounted and unsafe.";

    PathDiagnosticLocation BSLoc(V->getLocation(), BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(V->getSourceRange());
    BR->emitReport(std::move(Report));
  }
};
} // namespace

void ento::registerUncountedLocalVarsChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<UncountedLocalVarsChecker>();
}

bool ento::shouldRegisterUncountedLocalVarsChecker(const CheckerManager &) {
  return true;
}
