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
#include "clang/AST/DynamicRecursiveASTVisitor.h"
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
    struct LocalVisitor : DynamicRecursiveASTVisitor {
      const UncountedLambdaCapturesChecker *Checker;
      llvm::DenseSet<const DeclRefExpr *> DeclRefExprsToIgnore;
      llvm::DenseSet<const LambdaExpr *> LambdasToIgnore;
      QualType ClsType;

      explicit LocalVisitor(const UncountedLambdaCapturesChecker *Checker)
          : Checker(Checker) {
        assert(Checker);
        ShouldVisitTemplateInstantiations = true;
        ShouldVisitImplicitCode = false;
      }

      bool TraverseCXXMethodDecl(CXXMethodDecl *CXXMD) override {
        llvm::SaveAndRestore SavedDecl(ClsType);
        if (CXXMD->isInstance())
          ClsType = CXXMD->getThisType();
        return DynamicRecursiveASTVisitor::TraverseCXXMethodDecl(CXXMD);
      }

      bool shouldCheckThis() {
        auto result = !ClsType.isNull() ? isUnsafePtr(ClsType) : std::nullopt;
        return result && *result;
      }

      bool VisitLambdaExpr(LambdaExpr *L) override {
        if (LambdasToIgnore.contains(L))
          return true;
        Checker->visitLambdaExpr(L, shouldCheckThis());
        return true;
      }

      bool VisitVarDecl(VarDecl *VD) override {
        auto *Init = VD->getInit();
        if (!Init)
          return true;
        auto *L = dyn_cast_or_null<LambdaExpr>(Init->IgnoreParenCasts());
        if (!L)
          return true;
        LambdasToIgnore.insert(L); // Evaluate lambdas in VisitDeclRefExpr.
        return true;
      }

      bool VisitDeclRefExpr(DeclRefExpr *DRE) override {
        if (DeclRefExprsToIgnore.contains(DRE))
          return true;
        auto *VD = dyn_cast_or_null<VarDecl>(DRE->getDecl());
        if (!VD)
          return true;
        auto *Init = VD->getInit();
        if (!Init)
          return true;
        auto *L = dyn_cast_or_null<LambdaExpr>(Init->IgnoreParenCasts());
        if (!L)
          return true;
        LambdasToIgnore.insert(L);
        Checker->visitLambdaExpr(L, shouldCheckThis());
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

      bool VisitCallExpr(CallExpr *CE) override {
        checkCalleeLambda(CE);
        if (auto *Callee = CE->getDirectCallee()) {
          bool TreatAllArgsAsNoEscape = shouldTreatAllArgAsNoEscape(Callee);
          unsigned ArgIndex = 0;
          for (auto *Param : Callee->parameters()) {
            if (ArgIndex >= CE->getNumArgs())
              return true;
            auto *Arg = CE->getArg(ArgIndex)->IgnoreParenCasts();
            if (auto *L = findLambdaInArg(Arg)) {
              LambdasToIgnore.insert(L);
              if (!Param->hasAttr<NoEscapeAttr>() && !TreatAllArgsAsNoEscape)
                Checker->visitLambdaExpr(L, shouldCheckThis());
            }
            ++ArgIndex;
          }
        }
        return true;
      }

      LambdaExpr *findLambdaInArg(Expr *E) {
        if (auto *Lambda = dyn_cast_or_null<LambdaExpr>(E))
          return Lambda;
        auto *TempExpr = dyn_cast_or_null<CXXBindTemporaryExpr>(E);
        if (!TempExpr)
          return nullptr;
        E = TempExpr->getSubExpr()->IgnoreParenCasts();
        if (!E)
          return nullptr;
        if (auto *Lambda = dyn_cast<LambdaExpr>(E))
          return Lambda;
        auto *CE = dyn_cast_or_null<CXXConstructExpr>(E);
        if (!CE || !CE->getNumArgs())
          return nullptr;
        auto *CtorArg = CE->getArg(0)->IgnoreParenCasts();
        if (!CtorArg)
          return nullptr;
        if (auto *Lambda = dyn_cast<LambdaExpr>(CtorArg))
          return Lambda;
        auto *DRE = dyn_cast<DeclRefExpr>(CtorArg);
        if (!DRE)
          return nullptr;
        auto *VD = dyn_cast_or_null<VarDecl>(DRE->getDecl());
        if (!VD)
          return nullptr;
        auto *Init = VD->getInit();
        if (!Init)
          return nullptr;
        TempExpr = dyn_cast<CXXBindTemporaryExpr>(Init->IgnoreParenCasts());
        if (!TempExpr)
          return nullptr;
        return dyn_cast_or_null<LambdaExpr>(TempExpr->getSubExpr());
      }

      void checkCalleeLambda(CallExpr *CE) {
        auto *Callee = CE->getCallee();
        if (!Callee)
          return;
        auto *DRE = dyn_cast<DeclRefExpr>(Callee->IgnoreParenCasts());
        if (!DRE)
          return;
        auto *MD = dyn_cast_or_null<CXXMethodDecl>(DRE->getDecl());
        if (!MD || CE->getNumArgs() < 1)
          return;
        auto *Arg = CE->getArg(0)->IgnoreParenCasts();
        if (auto *L = dyn_cast_or_null<LambdaExpr>(Arg)) {
          LambdasToIgnore.insert(L); // Calling a lambda upon creation is safe.
          return;
        }
        auto *ArgRef = dyn_cast<DeclRefExpr>(Arg);
        if (!ArgRef)
          return;
        auto *VD = dyn_cast_or_null<VarDecl>(ArgRef->getDecl());
        if (!VD)
          return;
        auto *Init = VD->getInit();
        if (!Init)
          return;
        auto *L = dyn_cast_or_null<LambdaExpr>(Init->IgnoreParenCasts());
        if (!L)
          return;
        DeclRefExprsToIgnore.insert(ArgRef);
        LambdasToIgnore.insert(L);
        Checker->visitLambdaExpr(L, shouldCheckThis(),
                                 /* ignoreParamVarDecl */ true);
      }
    };

    LocalVisitor visitor(this);
    visitor.TraverseDecl(const_cast<TranslationUnitDecl *>(TUD));
  }

  void visitLambdaExpr(LambdaExpr *L, bool shouldCheckThis,
                       bool ignoreParamVarDecl = false) const {
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
      } else if (C.capturesThis() && shouldCheckThis) {
        if (ignoreParamVarDecl) // this is always a parameter to this function.
          continue;
        bool hasProtectThis = false;
        for (const LambdaCapture &OtherCapture : L->captures()) {
          if (!OtherCapture.capturesVariable())
            continue;
          if (auto *ValueDecl = OtherCapture.getCapturedVar()) {
            if (protectThis(ValueDecl)) {
              hasProtectThis = true;
              break;
            }
          }
        }
        if (!hasProtectThis)
          reportBugOnThisPtr(C);
      }
    }
  }

  bool protectThis(const ValueDecl *ValueDecl) const {
    auto *VD = dyn_cast<VarDecl>(ValueDecl);
    if (!VD)
      return false;
    auto *Init = VD->getInit()->IgnoreParenCasts();
    if (!Init)
      return false;
    auto *BTE = dyn_cast<CXXBindTemporaryExpr>(Init);
    if (!BTE)
      return false;
    auto *CE = dyn_cast_or_null<CXXConstructExpr>(BTE->getSubExpr());
    if (!CE)
      return false;
    auto *Ctor = CE->getConstructor();
    if (!Ctor)
      return false;
    auto clsName = safeGetName(Ctor->getParent());
    if (!isRefType(clsName) || !CE->getNumArgs())
      return false;
    auto *Arg = CE->getArg(0)->IgnoreParenCasts();
    while (auto *UO = dyn_cast<UnaryOperator>(Arg)) {
      auto OpCode = UO->getOpcode();
      if (OpCode == UO_Deref || OpCode == UO_AddrOf)
        Arg = UO->getSubExpr();
      else
        break;
    }
    return isa<CXXThisExpr>(Arg);
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
    Os << " to ref-counted type or CheckedPtr-capable type is unsafe.";

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

    Os << "raw-pointer 'this' to ref-counted type or CheckedPtr-capable type "
          "is unsafe.";

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
