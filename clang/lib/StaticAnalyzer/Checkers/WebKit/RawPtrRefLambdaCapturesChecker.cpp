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
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include <optional>

using namespace clang;
using namespace ento;

namespace {
class RawPtrRefLambdaCapturesChecker
    : public Checker<check::ASTDecl<TranslationUnitDecl>> {
private:
  BugType Bug;
  mutable BugReporter *BR = nullptr;
  TrivialFunctionAnalysis TFA;

protected:
  mutable std::optional<RetainTypeChecker> RTC;

public:
  RawPtrRefLambdaCapturesChecker(const char *description)
      : Bug(this, description, "WebKit coding guidelines") {}

  virtual std::optional<bool> isUnsafePtr(QualType) const = 0;
  virtual bool isPtrType(const std::string &) const = 0;
  virtual const char *ptrKind(QualType QT) const = 0;

  void checkASTDecl(const TranslationUnitDecl *TUD, AnalysisManager &MGR,
                    BugReporter &BRArg) const {
    BR = &BRArg;

    // The calls to checkAST* from AnalysisConsumer don't
    // visit template instantiations or lambda classes. We
    // want to visit those, so we make our own RecursiveASTVisitor.
    struct LocalVisitor : DynamicRecursiveASTVisitor {
      const RawPtrRefLambdaCapturesChecker *Checker;
      llvm::DenseSet<const DeclRefExpr *> DeclRefExprsToIgnore;
      llvm::DenseSet<const LambdaExpr *> LambdasToIgnore;
      llvm::DenseSet<const ValueDecl *> ProtectedThisDecls;
      llvm::DenseSet<const CXXConstructExpr *> ConstructToIgnore;

      QualType ClsType;

      explicit LocalVisitor(const RawPtrRefLambdaCapturesChecker *Checker)
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

      bool TraverseObjCMethodDecl(ObjCMethodDecl *OCMD) override {
        llvm::SaveAndRestore SavedDecl(ClsType);
        if (OCMD && OCMD->isInstanceMethod()) {
          if (auto *ImplParamDecl = OCMD->getSelfDecl())
            ClsType = ImplParamDecl->getType();
        }
        return DynamicRecursiveASTVisitor::TraverseObjCMethodDecl(OCMD);
      }

      bool VisitTypedefDecl(TypedefDecl *TD) override {
        if (Checker->RTC)
          Checker->RTC->visitTypedef(TD);
        return true;
      }

      bool shouldCheckThis() {
        auto result =
            !ClsType.isNull() ? Checker->isUnsafePtr(ClsType) : std::nullopt;
        return result && *result;
      }

      bool VisitLambdaExpr(LambdaExpr *L) override {
        if (LambdasToIgnore.contains(L))
          return true;
        Checker->visitLambdaExpr(L, shouldCheckThis() && !hasProtectedThis(L),
                                 ClsType);
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
        Checker->visitLambdaExpr(L, shouldCheckThis() && !hasProtectedThis(L),
                                 ClsType);
        return true;
      }

      bool shouldTreatAllArgAsNoEscape(FunctionDecl *Decl) {
        auto *NsDecl = Decl->getParent();
        if (!NsDecl || !isa<NamespaceDecl>(NsDecl))
          return false;
        // WTF::switchOn(T, F... f) is a variadic template function and couldn't
        // be annotated with NOESCAPE. We hard code it here to workaround that.
        if (safeGetName(NsDecl) == "WTF" && safeGetName(Decl) == "switchOn")
          return true;
        // Treat every argument of functions in std::ranges as noescape.
        if (safeGetName(NsDecl) == "ranges") {
          if (auto *OuterDecl = NsDecl->getParent();
              OuterDecl && isa<NamespaceDecl>(OuterDecl) &&
              safeGetName(OuterDecl) == "std")
            return true;
        }
        return false;
      }

      bool VisitCXXConstructExpr(CXXConstructExpr *CE) override {
        if (ConstructToIgnore.contains(CE))
          return true;
        if (auto *Callee = CE->getConstructor()) {
          unsigned ArgIndex = 0;
          for (auto *Param : Callee->parameters()) {
            if (ArgIndex >= CE->getNumArgs())
              return true;
            auto *Arg = CE->getArg(ArgIndex)->IgnoreParenCasts();
            if (auto *L = findLambdaInArg(Arg)) {
              LambdasToIgnore.insert(L);
              if (!Param->hasAttr<NoEscapeAttr>())
                Checker->visitLambdaExpr(
                    L, shouldCheckThis() && !hasProtectedThis(L), ClsType);
            }
            ++ArgIndex;
          }
        }
        return true;
      }

      bool VisitCallExpr(CallExpr *CE) override {
        checkCalleeLambda(CE);
        if (auto *Callee = CE->getDirectCallee()) {
          unsigned ArgIndex = isa<CXXOperatorCallExpr>(CE);
          bool TreatAllArgsAsNoEscape = shouldTreatAllArgAsNoEscape(Callee);
          for (auto *Param : Callee->parameters()) {
            if (ArgIndex >= CE->getNumArgs())
              return true;
            auto *Arg = CE->getArg(ArgIndex)->IgnoreParenCasts();
            if (auto *L = findLambdaInArg(Arg)) {
              LambdasToIgnore.insert(L);
              if (!Param->hasAttr<NoEscapeAttr>() && !TreatAllArgsAsNoEscape)
                Checker->visitLambdaExpr(
                    L, shouldCheckThis() && !hasProtectedThis(L), ClsType);
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
        auto *InnerCE = dyn_cast_or_null<CXXConstructExpr>(CtorArg);
        if (InnerCE && InnerCE->getNumArgs())
          CtorArg = InnerCE->getArg(0)->IgnoreParenCasts();
        auto updateIgnoreList = [&] {
          ConstructToIgnore.insert(CE);
          if (InnerCE)
            ConstructToIgnore.insert(InnerCE);
        };
        if (auto *Lambda = dyn_cast<LambdaExpr>(CtorArg)) {
          updateIgnoreList();
          return Lambda;
        }
        if (auto *TempExpr = dyn_cast<CXXBindTemporaryExpr>(CtorArg)) {
          E = TempExpr->getSubExpr()->IgnoreParenCasts();
          if (auto *Lambda = dyn_cast<LambdaExpr>(E)) {
            updateIgnoreList();
            return Lambda;
          }
        }
        auto *DRE = dyn_cast<DeclRefExpr>(CtorArg);
        if (!DRE)
          return nullptr;
        auto *VD = dyn_cast_or_null<VarDecl>(DRE->getDecl());
        if (!VD)
          return nullptr;
        auto *Init = VD->getInit();
        if (!Init)
          return nullptr;
        if (auto *Lambda = dyn_cast<LambdaExpr>(Init)) {
          DeclRefExprsToIgnore.insert(DRE);
          updateIgnoreList();
          return Lambda;
        }
        return nullptr;
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
      }

      bool hasProtectedThis(LambdaExpr *L) {
        for (const LambdaCapture &OtherCapture : L->captures()) {
          if (!OtherCapture.capturesVariable())
            continue;
          if (auto *ValueDecl = OtherCapture.getCapturedVar()) {
            if (declProtectsThis(ValueDecl)) {
              ProtectedThisDecls.insert(ValueDecl);
              return true;
            }
          }
        }
        return false;
      }

      bool declProtectsThis(const ValueDecl *ValueDecl) const {
        auto *VD = dyn_cast<VarDecl>(ValueDecl);
        if (!VD)
          return false;
        auto *Init = VD->getInit();
        if (!Init)
          return false;
        const Expr *Arg = Init->IgnoreParenCasts();
        do {
          if (auto *BTE = dyn_cast<CXXBindTemporaryExpr>(Arg))
            Arg = BTE->getSubExpr()->IgnoreParenCasts();
          if (auto *CE = dyn_cast<CXXConstructExpr>(Arg)) {
            auto *Ctor = CE->getConstructor();
            if (!Ctor)
              return false;
            auto clsName = safeGetName(Ctor->getParent());
            if (Checker->isPtrType(clsName) && CE->getNumArgs()) {
              Arg = CE->getArg(0)->IgnoreParenCasts();
              continue;
            }
            if (auto *Type = ClsType.getTypePtrOrNull()) {
              if (auto *CXXR = Type->getPointeeCXXRecordDecl()) {
                if (CXXR == Ctor->getParent() && Ctor->isMoveConstructor() &&
                    CE->getNumArgs() == 1) {
                  Arg = CE->getArg(0)->IgnoreParenCasts();
                  continue;
                }
              }
            }
            return false;
          }
          if (auto *CE = dyn_cast<CallExpr>(Arg)) {
            if (CE->isCallToStdMove() && CE->getNumArgs() == 1) {
              Arg = CE->getArg(0)->IgnoreParenCasts();
              continue;
            }
            if (auto *Callee = CE->getDirectCallee()) {
              if (isCtorOfSafePtr(Callee) && CE->getNumArgs() == 1) {
                Arg = CE->getArg(0)->IgnoreParenCasts();
                continue;
              }
            }
          }
          if (auto *OpCE = dyn_cast<CXXOperatorCallExpr>(Arg)) {
            auto OpCode = OpCE->getOperator();
            if (OpCode == OO_Star || OpCode == OO_Amp) {
              auto *Callee = OpCE->getDirectCallee();
              if (!Callee)
                return false;
              auto clsName = safeGetName(Callee->getParent());
              if (!Checker->isPtrType(clsName) || !OpCE->getNumArgs())
                return false;
              Arg = OpCE->getArg(0)->IgnoreParenCasts();
              continue;
            }
          }
          if (auto *UO = dyn_cast<UnaryOperator>(Arg)) {
            auto OpCode = UO->getOpcode();
            if (OpCode == UO_Deref || OpCode == UO_AddrOf) {
              Arg = UO->getSubExpr()->IgnoreParenCasts();
              continue;
            }
          }
          break;
        } while (Arg);
        if (auto *DRE = dyn_cast<DeclRefExpr>(Arg)) {
          auto *Decl = DRE->getDecl();
          if (auto *ImplicitParam = dyn_cast<ImplicitParamDecl>(Decl)) {
            auto kind = ImplicitParam->getParameterKind();
            return kind == ImplicitParamKind::ObjCSelf ||
                   kind == ImplicitParamKind::CXXThis;
          }
          return ProtectedThisDecls.contains(Decl);
        }
        return isa<CXXThisExpr>(Arg);
      }
    };

    LocalVisitor visitor(this);
    if (RTC)
      RTC->visitTranslationUnitDecl(TUD);
    visitor.TraverseDecl(const_cast<TranslationUnitDecl *>(TUD));
  }

  void visitLambdaExpr(LambdaExpr *L, bool shouldCheckThis, const QualType T,
                       bool ignoreParamVarDecl = false) const {
    if (TFA.isTrivial(L->getBody()))
      return;
    for (const LambdaCapture &C : L->captures()) {
      if (C.capturesVariable()) {
        ValueDecl *CapturedVar = C.getCapturedVar();
        if (ignoreParamVarDecl && isa<ParmVarDecl>(CapturedVar))
          continue;
        if (auto *ImplicitParam = dyn_cast<ImplicitParamDecl>(CapturedVar)) {
          auto kind = ImplicitParam->getParameterKind();
          if ((kind == ImplicitParamKind::ObjCSelf ||
               kind == ImplicitParamKind::CXXThis) &&
              !shouldCheckThis)
            continue;
        }
        QualType CapturedVarQualType = CapturedVar->getType();
        auto IsUncountedPtr = isUnsafePtr(CapturedVar->getType());
        if (C.getCaptureKind() == LCK_ByCopy &&
            CapturedVarQualType->isReferenceType())
          continue;
        if (IsUncountedPtr && *IsUncountedPtr)
          reportBug(C, CapturedVar, CapturedVarQualType, L);
      } else if (C.capturesThis() && shouldCheckThis) {
        if (ignoreParamVarDecl) // this is always a parameter to this function.
          continue;
        reportBugOnThisPtr(C, T);
      }
    }
  }

  void reportBug(const LambdaCapture &Capture, ValueDecl *CapturedVar,
                 const QualType T, LambdaExpr *L) const {
    assert(CapturedVar);

    auto Location = Capture.getLocation();
    if (isa<ImplicitParamDecl>(CapturedVar) && !Location.isValid())
      Location = L->getBeginLoc();

    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    if (Capture.isExplicit()) {
      Os << "Captured ";
    } else {
      Os << "Implicitly captured ";
    }
    if (isa<PointerType>(T) || isa<ObjCObjectPointerType>(T)) {
      Os << "raw-pointer ";
    } else {
      Os << "reference ";
    }

    printQuotedQualifiedName(Os, CapturedVar);
    Os << " to " << ptrKind(T) << " type is unsafe.";

    PathDiagnosticLocation BSLoc(Location, BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    BR->emitReport(std::move(Report));
  }

  void reportBugOnThisPtr(const LambdaCapture &Capture,
                          const QualType T) const {
    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    if (Capture.isExplicit()) {
      Os << "Captured ";
    } else {
      Os << "Implicitly captured ";
    }

    Os << "raw-pointer 'this' to " << ptrKind(T) << " type is unsafe.";

    PathDiagnosticLocation BSLoc(Capture.getLocation(), BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    BR->emitReport(std::move(Report));
  }
};

class UncountedLambdaCapturesChecker : public RawPtrRefLambdaCapturesChecker {
public:
  UncountedLambdaCapturesChecker()
      : RawPtrRefLambdaCapturesChecker("Lambda capture of uncounted or "
                                       "unchecked variable") {}

  std::optional<bool> isUnsafePtr(QualType QT) const final {
    auto result1 = isUncountedPtr(QT);
    auto result2 = isUncheckedPtr(QT);
    if (result1 && *result1)
      return true;
    if (result2 && *result2)
      return true;
    if (result1)
      return *result1;
    return result2;
  }

  virtual bool isPtrType(const std::string &Name) const final {
    return isRefType(Name) || isCheckedPtr(Name);
  }

  const char *ptrKind(QualType QT) const final {
    if (isUncounted(QT))
      return "uncounted";
    return "unchecked";
  }
};

class UnretainedLambdaCapturesChecker : public RawPtrRefLambdaCapturesChecker {
public:
  UnretainedLambdaCapturesChecker()
      : RawPtrRefLambdaCapturesChecker("Lambda capture of unretained "
                                       "variables") {
    RTC = RetainTypeChecker();
  }

  std::optional<bool> isUnsafePtr(QualType QT) const final {
    return RTC->isUnretained(QT);
  }

  virtual bool isPtrType(const std::string &Name) const final {
    return isRetainPtr(Name);
  }

  const char *ptrKind(QualType QT) const final { return "unretained"; }
};

} // namespace

void ento::registerUncountedLambdaCapturesChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<UncountedLambdaCapturesChecker>();
}

bool ento::shouldRegisterUncountedLambdaCapturesChecker(
    const CheckerManager &mgr) {
  return true;
}

void ento::registerUnretainedLambdaCapturesChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<UnretainedLambdaCapturesChecker>();
}

bool ento::shouldRegisterUnretainedLambdaCapturesChecker(
    const CheckerManager &mgr) {
  return true;
}
