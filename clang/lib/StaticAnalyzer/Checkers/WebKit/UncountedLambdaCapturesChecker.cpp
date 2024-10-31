//=======- UncountedLambdaCapturesChecker.cpp --------------------*- C++ -*-==//
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
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include <optional>

using namespace clang;
using namespace ento;

namespace {
class UncountedLambdaCapturesChecker
    : public Checker<check::ASTDecl<TranslationUnitDecl>> {
private:
  BugType Bug{this, "Lambda capture of uncounted variable",
              "WebKit coding guidelines"};
  mutable BugReporter *BR = nullptr;
  TrivialFunctionAnalysis TFA;

public:
  void checkASTDecl(const TranslationUnitDecl *TUD, AnalysisManager &MGR,
                    BugReporter &BRArg) const {
    BR = &BRArg;

    // The calls to checkAST* from AnalysisConsumer don't
    // visit template instantiations or lambda classes. We
    // want to visit those, so we make our own RecursiveASTVisitor.
    struct LocalVisitor : public RecursiveASTVisitor<LocalVisitor> {
      const UncountedLambdaCapturesChecker *Checker;
      llvm::DenseSet<const DeclRefExpr *> DeclRefExprsToIgnore;

      explicit LocalVisitor(const UncountedLambdaCapturesChecker *Checker)
          : Checker(Checker) {
        assert(Checker);
      }

      bool shouldVisitTemplateInstantiations() const { return true; }
      bool shouldVisitImplicitCode() const { return false; }

      bool VisitDeclRefExpr(DeclRefExpr *DRE) {
        if (DeclRefExprsToIgnore.contains(DRE))
          return true;
        auto *VD = dyn_cast_or_null<VarDecl>(DRE->getDecl());
        if (!VD)
          return true;
        auto *Init = VD->getInit()->IgnoreParenCasts();
        auto *L = dyn_cast_or_null<LambdaExpr>(Init);
        if (!L)
          return true;
        Checker->visitLambdaExpr(L);
        return true;
      }

      // WTF::switchOn(T, F... f) is a variadic template function and couldn't
      // be annotated with NOESCAPE. We hard code it here to workaround that.
      bool shouldTreatAllArgAsNoEscape(FunctionDecl *Decl) {
        auto *NsDecl = Decl->getParent();
        if (!NsDecl || !isa<NamespaceDecl>(NsDecl))
          return false;
        return safeGetName(NsDecl) == "WTF" && safeGetName(Decl) == "switchOn";
      }

      bool VisitCallExpr(CallExpr *CE) {
        checkCalleeLambda(CE);
        if (auto *Callee = CE->getDirectCallee()) {
          bool TreatAllArgsAsNoEscape = shouldTreatAllArgAsNoEscape(Callee);
          unsigned ArgIndex = 0;
          for (auto *Param : Callee->parameters()) {
            if (ArgIndex >= CE->getNumArgs())
              break;
            auto *Arg = CE->getArg(ArgIndex)->IgnoreParenCasts();
            if (!Param->hasAttr<NoEscapeAttr>() && !TreatAllArgsAsNoEscape) {
              if (auto *L = dyn_cast_or_null<LambdaExpr>(Arg))
                Checker->visitLambdaExpr(L);
            }
            ++ArgIndex;
          }
        }
        return true;
      }

      void checkCalleeLambda(CallExpr *CE) {
        auto *Callee = CE->getCallee();
        if (!Callee)
          return;
        auto *DRE = dyn_cast<DeclRefExpr>(Callee->IgnoreParenCasts());
        if (!DRE)
          return;
        auto *MD = dyn_cast_or_null<CXXMethodDecl>(DRE->getDecl());
        if (!MD || CE->getNumArgs() != 1)
          return;
        auto *Arg = CE->getArg(0)->IgnoreParenCasts();
        auto *ArgRef = dyn_cast<DeclRefExpr>(Arg);
        if (!ArgRef)
          return;
        auto *VD = dyn_cast_or_null<VarDecl>(ArgRef->getDecl());
        if (!VD)
          return;
        auto *Init = VD->getInit()->IgnoreParenCasts();
        auto *L = dyn_cast_or_null<LambdaExpr>(Init);
        if (!L)
          return;
        DeclRefExprsToIgnore.insert(ArgRef);
        Checker->visitLambdaExpr(L, /* ignoreParamVarDecl */ true);
      }
    };

    LocalVisitor visitor(this);
    visitor.TraverseDecl(const_cast<TranslationUnitDecl *>(TUD));
  }

  void visitLambdaExpr(LambdaExpr *L, bool ignoreParamVarDecl = false) const {
    if (TFA.isTrivial(L->getBody()))
      return;
    for (const LambdaCapture &C : L->captures()) {
      if (C.capturesVariable()) {
        ValueDecl *CapturedVar = C.getCapturedVar();
        if (ignoreParamVarDecl && isa<ParmVarDecl>(CapturedVar))
          continue;
        QualType CapturedVarQualType = CapturedVar->getType();
        auto IsUncountedPtr = isUnsafePtr(CapturedVar->getType());
        if (IsUncountedPtr && *IsUncountedPtr)
          reportBug(C, CapturedVar, CapturedVarQualType);
      } else if (C.capturesThis()) {
        if (ignoreParamVarDecl) // this is always a parameter to this function.
          continue;
        reportBugOnThisPtr(C);
      }
    }
  }

  void reportBug(const LambdaCapture &Capture, ValueDecl *CapturedVar,
                 const QualType T) const {
    assert(CapturedVar);

    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    if (Capture.isExplicit()) {
      Os << "Captured ";
    } else {
      Os << "Implicitly captured ";
    }
    if (T->isPointerType()) {
      Os << "raw-pointer ";
    } else {
      assert(T->isReferenceType());
      Os << "reference ";
    }

    printQuotedQualifiedName(Os, Capture.getCapturedVar());
    Os << " to ref-counted / CheckedPtr capable type is unsafe.";

    PathDiagnosticLocation BSLoc(Capture.getLocation(), BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    BR->emitReport(std::move(Report));
  }

  void reportBugOnThisPtr(const LambdaCapture &Capture) const {
    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    if (Capture.isExplicit()) {
      Os << "Captured ";
    } else {
      Os << "Implicitly captured ";
    }

    Os << "raw-pointer 'this' to ref-counted / CheckedPtr capable type is "
          "unsafe.";

    PathDiagnosticLocation BSLoc(Capture.getLocation(), BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    BR->emitReport(std::move(Report));
  }
};
} // namespace

void ento::registerUncountedLambdaCapturesChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<UncountedLambdaCapturesChecker>();
}

bool ento::shouldRegisterUncountedLambdaCapturesChecker(
    const CheckerManager &mgr) {
  return true;
}
