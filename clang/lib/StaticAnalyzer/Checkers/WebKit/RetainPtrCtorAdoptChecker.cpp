//=======- RetainPtrCtorAdoptChecker.cpp -------------------------*- C++ -*-==//
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
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/RetainSummaryManager.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

using namespace clang;
using namespace ento;

namespace {

class RetainPtrCtorAdoptChecker
    : public Checker<check::ASTDecl<TranslationUnitDecl>> {
private:
  BugType Bug;
  mutable BugReporter *BR;
  mutable std::unique_ptr<RetainSummaryManager> Summaries;
  mutable llvm::DenseSet<const ValueDecl *> CreateOrCopyOutArguments;
  mutable RetainTypeChecker RTC;

public:
  RetainPtrCtorAdoptChecker()
      : Bug(this, "Correct use of RetainPtr, adoptNS, and adoptCF",
            "WebKit coding guidelines") {}

  void checkASTDecl(const TranslationUnitDecl *TUD, AnalysisManager &MGR,
                    BugReporter &BRArg) const {
    BR = &BRArg;

    // The calls to checkAST* from AnalysisConsumer don't
    // visit template instantiations or lambda classes. We
    // want to visit those, so we make our own RecursiveASTVisitor.
    struct LocalVisitor : public RecursiveASTVisitor<LocalVisitor> {
      const RetainPtrCtorAdoptChecker *Checker;
      Decl *DeclWithIssue{nullptr};

      using Base = RecursiveASTVisitor<LocalVisitor>;

      explicit LocalVisitor(const RetainPtrCtorAdoptChecker *Checker)
          : Checker(Checker) {
        assert(Checker);
      }

      bool shouldVisitTemplateInstantiations() const { return true; }
      bool shouldVisitImplicitCode() const { return false; }

      bool TraverseDecl(Decl *D) {
        llvm::SaveAndRestore SavedDecl(DeclWithIssue);
        if (D && (isa<FunctionDecl>(D) || isa<ObjCMethodDecl>(D)))
          DeclWithIssue = D;
        return Base::TraverseDecl(D);
      }

      bool TraverseClassTemplateDecl(ClassTemplateDecl *CTD) {
        if (safeGetName(CTD) == "RetainPtr")
          return true; // Skip the contents of RetainPtr.
        return Base::TraverseClassTemplateDecl(CTD);
      }

      bool VisitTypedefDecl(TypedefDecl *TD) {
        Checker->RTC.visitTypedef(TD);
        return true;
      }

      bool VisitCallExpr(const CallExpr *CE) {
        Checker->visitCallExpr(CE, DeclWithIssue);
        return true;
      }

      bool VisitCXXConstructExpr(const CXXConstructExpr *CE) {
        Checker->visitConstructExpr(CE, DeclWithIssue);
        return true;
      }
    };

    LocalVisitor visitor(this);
    Summaries = std::make_unique<RetainSummaryManager>(
        TUD->getASTContext(), true /* trackObjCAndCFObjects */,
        false /* trackOSObjects */);
    RTC.visitTranslationUnitDecl(TUD);
    visitor.TraverseDecl(const_cast<TranslationUnitDecl *>(TUD));
  }

  bool isAdoptFn(const Decl *FnDecl) const {
    auto Name = safeGetName(FnDecl);
    return Name == "adoptNS" || Name == "adoptCF" || Name == "adoptNSArc" ||
           Name == "adoptCFArc";
  }

  bool isAdoptNS(const Decl *FnDecl) const {
    auto Name = safeGetName(FnDecl);
    return Name == "adoptNS" || Name == "adoptNSArc";
  }

  void visitCallExpr(const CallExpr *CE, const Decl *DeclWithIssue) const {
    if (BR->getSourceManager().isInSystemHeader(CE->getExprLoc()))
      return;

    auto *F = CE->getDirectCallee();
    if (!F)
      return;

    if (!isAdoptFn(F) || !CE->getNumArgs()) {
      rememberOutArguments(CE, F);
      return;
    }

    auto *Arg = CE->getArg(0)->IgnoreParenCasts();
    auto Result = isOwned(Arg);
    auto Name = safeGetName(F);
    if (Result == IsOwnedResult::Unknown)
      Result = IsOwnedResult::NotOwned;
    if (Result == IsOwnedResult::NotOwned && !isAllocInit(Arg) &&
        !isCreateOrCopy(Arg)) {
      if (auto *DRE = dyn_cast<DeclRefExpr>(Arg)) {
        if (CreateOrCopyOutArguments.contains(DRE->getDecl()))
          return;
      }
      if (RTC.isARCEnabled() && isAdoptNS(F))
        reportUseAfterFree(Name, CE, DeclWithIssue, "when ARC is disabled");
      else
        reportUseAfterFree(Name, CE, DeclWithIssue);
    }
  }

  void rememberOutArguments(const CallExpr *CE,
                            const FunctionDecl *Callee) const {
    if (!isCreateOrCopyFunction(Callee))
      return;

    unsigned ArgCount = CE->getNumArgs();
    for (unsigned ArgIndex = 0; ArgIndex < ArgCount; ++ArgIndex) {
      auto *Arg = CE->getArg(ArgIndex)->IgnoreParenCasts();
      auto *Unary = dyn_cast<UnaryOperator>(Arg);
      if (!Unary)
        continue;
      if (Unary->getOpcode() != UO_AddrOf)
        continue;
      auto *SubExpr = Unary->getSubExpr();
      if (!SubExpr)
        continue;
      auto *DRE = dyn_cast<DeclRefExpr>(SubExpr->IgnoreParenCasts());
      if (!DRE)
        continue;
      auto *Decl = DRE->getDecl();
      if (!Decl)
        continue;
      CreateOrCopyOutArguments.insert(Decl);
    }
  }

  void visitConstructExpr(const CXXConstructExpr *CE,
                          const Decl *DeclWithIssue) const {
    if (BR->getSourceManager().isInSystemHeader(CE->getExprLoc()))
      return;

    auto *Ctor = CE->getConstructor();
    if (!Ctor)
      return;

    auto *Cls = Ctor->getParent();
    if (!Cls)
      return;

    if (safeGetName(Cls) != "RetainPtr" || !CE->getNumArgs())
      return;

    // Ignore RetainPtr construction inside adoptNS, adoptCF, and retainPtr.
    if (isAdoptFn(DeclWithIssue) || safeGetName(DeclWithIssue) == "retainPtr")
      return;

    std::string Name = "RetainPtr constructor";
    auto *Arg = CE->getArg(0)->IgnoreParenCasts();
    auto Result = isOwned(Arg);
    if (Result == IsOwnedResult::Unknown)
      Result = IsOwnedResult::NotOwned;
    if (Result == IsOwnedResult::Owned)
      reportLeak(Name, CE, DeclWithIssue);
    else if (RTC.isARCEnabled() && isAllocInit(Arg))
      reportLeak(Name, CE, DeclWithIssue, "when ARC is disabled");
    else if (isCreateOrCopy(Arg))
      reportLeak(Name, CE, DeclWithIssue);
  }

  bool isAllocInit(const Expr *E) const {
    auto *ObjCMsgExpr = dyn_cast<ObjCMessageExpr>(E);
    if (!ObjCMsgExpr)
      return false;
    auto Selector = ObjCMsgExpr->getSelector();
    auto NameForFirstSlot = Selector.getNameForSlot(0);
    if (NameForFirstSlot == "alloc" || NameForFirstSlot.starts_with("copy") ||
        NameForFirstSlot.starts_with("mutableCopy"))
      return true;
    if (!NameForFirstSlot.starts_with("init"))
      return false;
    if (!ObjCMsgExpr->isInstanceMessage())
      return false;
    auto *Receiver = ObjCMsgExpr->getInstanceReceiver()->IgnoreParenCasts();
    if (!Receiver)
      return false;
    if (auto *InnerObjCMsgExpr = dyn_cast<ObjCMessageExpr>(Receiver)) {
      auto InnerSelector = InnerObjCMsgExpr->getSelector();
      return InnerSelector.getNameForSlot(0) == "alloc";
    } else if (auto *CE = dyn_cast<CallExpr>(Receiver)) {
      if (auto *Callee = CE->getDirectCallee()) {
        auto CalleeName = Callee->getName();
        return CalleeName.starts_with("alloc");
      }
    }
    return false;
  }

  bool isCreateOrCopy(const Expr *E) const {
    auto *CE = dyn_cast<CallExpr>(E);
    if (!CE)
      return false;
    auto *Callee = CE->getDirectCallee();
    if (!Callee)
      return false;
    return isCreateOrCopyFunction(Callee);
  }

  bool isCreateOrCopyFunction(const FunctionDecl *FnDecl) const {
    auto CalleeName = safeGetName(FnDecl);
    return CalleeName.find("Create") != std::string::npos ||
           CalleeName.find("Copy") != std::string::npos;
  }

  enum class IsOwnedResult { Unknown, Skip, Owned, NotOwned };
  IsOwnedResult isOwned(const Expr *E) const {
    while (1) {
      if (isa<CXXNullPtrLiteralExpr>(E))
        return IsOwnedResult::NotOwned;
      if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
        auto QT = DRE->getType();
        if (isRetainPtrType(QT))
          return IsOwnedResult::NotOwned;
        QT = QT.getCanonicalType();
        if (RTC.isUnretained(QT, true /* ignoreARC */))
          return IsOwnedResult::NotOwned;
        auto *PointeeType = QT->getPointeeType().getTypePtrOrNull();
        if (PointeeType && PointeeType->isVoidType())
          return IsOwnedResult::NotOwned; // Assume reading void* as +0.
      }
      if (auto *TE = dyn_cast<CXXBindTemporaryExpr>(E)) {
        E = TE->getSubExpr();
        continue;
      }
      if (auto *ObjCMsgExpr = dyn_cast<ObjCMessageExpr>(E)) {
        auto Summary = Summaries->getSummary(AnyCall(ObjCMsgExpr));
        auto RetEffect = Summary->getRetEffect();
        switch (RetEffect.getKind()) {
        case RetEffect::NoRet:
          return IsOwnedResult::Unknown;
        case RetEffect::OwnedSymbol:
          return IsOwnedResult::Owned;
        case RetEffect::NotOwnedSymbol:
          return IsOwnedResult::NotOwned;
        case RetEffect::OwnedWhenTrackedReceiver:
          if (auto *Receiver = ObjCMsgExpr->getInstanceReceiver()) {
            E = Receiver->IgnoreParenCasts();
            continue;
          }
          return IsOwnedResult::Unknown;
        case RetEffect::NoRetHard:
          return IsOwnedResult::Unknown;
        }
      }
      if (auto *CXXCE = dyn_cast<CXXMemberCallExpr>(E)) {
        if (auto *MD = CXXCE->getMethodDecl()) {
          auto *Cls = MD->getParent();
          if (auto *CD = dyn_cast<CXXConversionDecl>(MD)) {
            auto QT = CD->getConversionType().getCanonicalType();
            auto *ResultType = QT.getTypePtrOrNull();
            if (safeGetName(Cls) == "RetainPtr" && ResultType &&
                (ResultType->isPointerType() || ResultType->isReferenceType() ||
                 ResultType->isObjCObjectPointerType()))
              return IsOwnedResult::NotOwned;
          }
          if (safeGetName(MD) == "leakRef" && safeGetName(Cls) == "RetainPtr")
            return IsOwnedResult::Owned;
        }
      }
      if (auto *CE = dyn_cast<CallExpr>(E)) {
        if (auto *Callee = CE->getDirectCallee()) {
          if (isAdoptFn(Callee))
            return IsOwnedResult::NotOwned;
          if (safeGetName(Callee) == "__builtin___CFStringMakeConstantString")
            return IsOwnedResult::NotOwned;
          auto RetType = Callee->getReturnType();
          if (isRetainPtrType(RetType))
            return IsOwnedResult::NotOwned;
        } else if (auto *CalleeExpr = CE->getCallee()) {
          if (isa<CXXDependentScopeMemberExpr>(CalleeExpr))
            return IsOwnedResult::Skip; // Wait for instantiation.
        }
        auto Summary = Summaries->getSummary(AnyCall(CE));
        auto RetEffect = Summary->getRetEffect();
        switch (RetEffect.getKind()) {
        case RetEffect::NoRet:
          return IsOwnedResult::Unknown;
        case RetEffect::OwnedSymbol:
          return IsOwnedResult::Owned;
        case RetEffect::NotOwnedSymbol:
          return IsOwnedResult::NotOwned;
        case RetEffect::OwnedWhenTrackedReceiver:
          return IsOwnedResult::Unknown;
        case RetEffect::NoRetHard:
          return IsOwnedResult::Unknown;
        }
      }
      break;
    }
    return IsOwnedResult::Unknown;
  }

  void reportUseAfterFree(std::string &Name, const CallExpr *CE,
                          const Decl *DeclWithIssue,
                          const char *condition = nullptr) const {
    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    Os << "Incorrect use of " << Name
       << ". The argument is +0 and results in an use-after-free";
    if (condition)
      Os << " " << condition;
    Os << ".";

    PathDiagnosticLocation BSLoc(CE->getSourceRange().getBegin(),
                                 BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(CE->getSourceRange());
    Report->setDeclWithIssue(DeclWithIssue);
    BR->emitReport(std::move(Report));
  }

  void reportLeak(std::string &Name, const CXXConstructExpr *CE,
                  const Decl *DeclWithIssue,
                  const char *condition = nullptr) const {
    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    Os << "Incorrect use of " << Name
       << ". The argument is +1 and results in a memory leak";
    if (condition)
      Os << " " << condition;
    Os << ".";

    PathDiagnosticLocation BSLoc(CE->getSourceRange().getBegin(),
                                 BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(CE->getSourceRange());
    Report->setDeclWithIssue(DeclWithIssue);
    BR->emitReport(std::move(Report));
  }
};
} // namespace

void ento::registerRetainPtrCtorAdoptChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<RetainPtrCtorAdoptChecker>();
}

bool ento::shouldRegisterRetainPtrCtorAdoptChecker(const CheckerManager &mgr) {
  return true;
}
