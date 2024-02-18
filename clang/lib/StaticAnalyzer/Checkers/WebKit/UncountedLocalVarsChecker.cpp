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

// for ( int a = ...) ... true
// for ( int a : ...) ... true
// if ( int* a = ) ... true
// anything else ... false
bool isDeclaredInForOrIf(const VarDecl *Var) {
  assert(Var);
  auto &ASTCtx = Var->getASTContext();
  auto parent = ASTCtx.getParents(*Var);

  if (parent.size() == 1) {
    if (auto *DS = parent.begin()->get<DeclStmt>()) {
      DynTypedNodeList grandParent = ASTCtx.getParents(*DS);
      if (grandParent.size() == 1) {
        return grandParent.begin()->get<ForStmt>() ||
               grandParent.begin()->get<IfStmt>() ||
               grandParent.begin()->get<CXXForRangeStmt>();
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
      const Expr *const InitExpr = V->getInit();
      if (!InitExpr)
        return; // FIXME: later on we might warn on uninitialized vars too

      const clang::Expr *const InitArgOrigin =
          tryToFindPtrOrigin(InitExpr, /*StopAtFirstRefCountedObj=*/false)
              .first;
      if (!InitArgOrigin)
        return;

      if (isVarDeclGuardedInit(V, InitArgOrigin))
        return;

      reportBug(V);
    }
  }

  bool shouldSkipVarDecl(const VarDecl *V) const {
    assert(V);
    if (!V->isLocalVarDecl())
      return true;

    if (isDeclaredInForOrIf(V))
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
