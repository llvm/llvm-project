//=======- RetainPtrCtorAdoptChecker.cpp -------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTUtils.h"
#include "PtrTypesSemantics.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/DomainSpecific/CocoaConventions.h"
#include "clang/Analysis/RetainSummaryManager.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

using namespace clang;
using namespace ento;

namespace {

class RetainPtrCtorAdoptChecker
    : public Checker<check::ASTDecl<TranslationUnitDecl>> {
private:
  BugType Bug;
  mutable BugReporter *BR = nullptr;
  mutable std::unique_ptr<RetainSummaryManager> Summaries;
  mutable llvm::DenseSet<const ValueDecl *> CreateOrCopyOutArguments;
  mutable llvm::DenseSet<const Expr *> CreateOrCopyFnCall;
  // Leaks are reported once the enclosing function body has been traversed
  // since an expression may be marked as consumed after it has been visited,
  // e.g. by a return statement which follows it.
  mutable SmallVector<std::pair<const Expr *, const Decl *>, 16> PendingLeaks;
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
        // Whether D is a function or a method which isn't nested in another
        // one, e.g. via a lambda or a local class.
        bool IsOutermostCallable = false;
        if (D && (isa<FunctionDecl>(D) || isa<ObjCMethodDecl>(D))) {
          IsOutermostCallable = !DeclWithIssue;
          DeclWithIssue = D;
        }
        bool Result = Base::TraverseDecl(D);
        // Every +1 value in a function body is consumed by that same body or
        // else leaked, so the pending reports can be flushed here instead of
        // being accumulated for the entire translation unit. Nested callables
        // must not flush since a return statement later in the enclosing body
        // may still consume a value which precedes them.
        if (IsOutermostCallable)
          Checker->reportPendingLeaks();
        return Result;
      }

      bool TraverseClassTemplateDecl(ClassTemplateDecl *CTD) {
        if (isRetainPtrOrOSPtr(safeGetName(CTD)))
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

      bool VisitObjCMessageExpr(const ObjCMessageExpr *ObjCMsgExpr) {
        Checker->visitObjCMessageExpr(ObjCMsgExpr, DeclWithIssue);
        return true;
      }

      bool VisitReturnStmt(const ReturnStmt *RS) {
        Checker->visitReturnStmt(RS, DeclWithIssue);
        return true;
      }

      bool VisitVarDecl(const VarDecl *VD) {
        Checker->visitVarDecl(VD);
        return true;
      }

      bool VisitBinaryOperator(const BinaryOperator *BO) {
        Checker->visitBinaryOperator(BO);
        return true;
      }
    };

    LocalVisitor visitor(this);
    Summaries = std::make_unique<RetainSummaryManager>(
        TUD->getASTContext(), true /* trackObjCAndCFObjects */,
        false /* trackOSObjects */);
    RTC.visitTranslationUnitDecl(TUD);
    visitor.TraverseDecl(const_cast<TranslationUnitDecl *>(TUD));

    // Flush whatever was found outside of a function body such as a global
    // variable's initializer.
    reportPendingLeaks();
  }

  // Reports the leaks found in the function body which has just been traversed
  // and discards the state which only pertains to that body.
  void reportPendingLeaks() const {
    for (auto &[E, DeclWithIssue] : PendingLeaks) {
      if (!CreateOrCopyFnCall.contains(E))
        reportLeak(E, DeclWithIssue);
    }
    PendingLeaks.clear();
    CreateOrCopyFnCall.clear();
    CreateOrCopyOutArguments.clear();
  }

  bool isAdoptFn(const Decl *FnDecl) const {
    return isAdoptFnName(safeGetName(FnDecl));
  }

  bool isAdoptFnName(const std::string &Name) const {
    return isAdoptNS(Name) || Name == "adoptCF" || Name == "adoptCFArc" ||
           Name == "adoptCFNullable" || Name == "adoptCFNullableArc" ||
           Name == "adoptOSObject" || Name == "adoptOSObjectArc";
  }

  bool isAdoptNS(const std::string &Name) const {
    return Name == "adoptNS" || Name == "adoptNSArc" ||
           Name == "adoptNSNullable" || Name == "adoptNSNullableArc";
  }

  void visitCallExpr(const CallExpr *CE, const Decl *DeclWithIssue) const {
    assert(BR && "expected nonnull BugReporter");
    if (BR->getSourceManager().isInSystemHeader(CE->getExprLoc()))
      return;

    std::string FnName;
    if (auto *F = CE->getDirectCallee()) {
      FnName = safeGetName(F);
      if (isAdoptFnName(FnName))
        checkAdoptCall(CE, FnName, DeclWithIssue);
      else {
        checkCreateOrCopyFunction(CE, DeclWithIssue);
        checkBridgingRelease(CE, F, DeclWithIssue);
      }
      return;
    }

    auto *CalleeExpr = CE->getCallee();
    if (!CalleeExpr)
      return;
    CalleeExpr = CalleeExpr->IgnoreParenCasts();
    if (auto *UnresolvedExpr = dyn_cast<UnresolvedLookupExpr>(CalleeExpr)) {
      auto Name = UnresolvedExpr->getName();
      if (!Name.isIdentifier())
        return;
      FnName = Name.getAsString();
      if (isAdoptFnName(FnName))
        checkAdoptCall(CE, FnName, DeclWithIssue);
    }
    checkCreateOrCopyFunction(CE, DeclWithIssue);
  }

  void checkAdoptCall(const CallExpr *CE, const std::string &FnName,
                      const Decl *DeclWithIssue) const {
    if (!CE->getNumArgs())
      return;

    auto *Arg = CE->getArg(0)->IgnoreParenCasts();
    auto Result = isOwned(Arg);
    if (Result == IsOwnedResult::Unknown)
      Result = IsOwnedResult::NotOwned;

    const Expr *Inner = nullptr;
    if (isAllocInit(Arg, &Inner) || isCreateOrCopy(Arg)) {
      if (Inner)
        CreateOrCopyFnCall.insert(Inner);
      CreateOrCopyFnCall.insert(Arg); // Avoid double reporting.
      return;
    }
    if (Result == IsOwnedResult::Owned || Result == IsOwnedResult::Skip ||
        isNullPtr(Arg)) {
      CreateOrCopyFnCall.insert(Arg);
      return;
    }

    if (auto *DRE = dyn_cast<DeclRefExpr>(Arg)) {
      if (CreateOrCopyOutArguments.contains(DRE->getDecl()))
        return;
    }
    if (RTC.isARCEnabled() && isAdoptFnName(FnName))
      reportUseAfterFree(FnName, CE, DeclWithIssue, "when ARC is disabled");
    else
      reportUseAfterFree(FnName, CE, DeclWithIssue);
  }

  void visitObjCMessageExpr(const ObjCMessageExpr *ObjCMsgExpr,
                            const Decl *DeclWithIssue) const {
    if (BR->getSourceManager().isInSystemHeader(ObjCMsgExpr->getExprLoc()))
      return;

    auto Selector = ObjCMsgExpr->getSelector();
    if (Selector.getAsString() == "autorelease") {
      auto *Receiver = ObjCMsgExpr->getInstanceReceiver()->IgnoreParenCasts();
      if (!Receiver)
        return;
      ObjCMsgExpr = dyn_cast<ObjCMessageExpr>(Receiver);
      if (!ObjCMsgExpr)
        return;
      const Expr *Inner = nullptr;
      if (!isAllocInit(ObjCMsgExpr, &Inner))
        return;
      CreateOrCopyFnCall.insert(ObjCMsgExpr);
      if (Inner)
        CreateOrCopyFnCall.insert(Inner);
      return;
    }

    const Expr *Inner = nullptr;
    if (!isAllocInit(ObjCMsgExpr, &Inner))
      return;
    if (RTC.isARCEnabled())
      return; // ARC never leaks.
    if (Inner)
      CreateOrCopyFnCall.insert(Inner); // Avoid double reporting.
    PendingLeaks.emplace_back(ObjCMsgExpr, DeclWithIssue);
  }

  void checkCreateOrCopyFunction(const CallExpr *CE,
                                 const Decl *DeclWithIssue) const {
    unsigned ArgCount = CE->getNumArgs();
    auto *CalleeDecl = CE->getCalleeDecl();
    auto *FnDecl = CalleeDecl ? CalleeDecl->getAsFunction() : nullptr;
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
      if (FnDecl && ArgIndex < FnDecl->getNumParams()) {
        // Manually check attributes on argumenet since RetainSummaryManager
        // basically ignores CF_RETRUNS_RETAINED on out arguments.
        auto *ParamDecl = FnDecl->getParamDecl(ArgIndex);
        if (ParamDecl->hasAttr<CFReturnsRetainedAttr>())
          CreateOrCopyOutArguments.insert(Decl);
      } else {
        // No callee or a variadic argument.
        // Conservatively assume it's an out argument.
        if (RTC.isUnretained(Decl->getType()))
          CreateOrCopyOutArguments.insert(Decl);
      }
    }
    auto Summary = Summaries->getSummary(AnyCall(CE));
    switch (Summary->getRetEffect().getKind()) {
    case RetEffect::OwnedSymbol:
    case RetEffect::OwnedWhenTrackedReceiver:
      PendingLeaks.emplace_back(CE, DeclWithIssue);
      break;
    default:
      break;
    }
  }

  void checkBridgingRelease(const CallExpr *CE, const FunctionDecl *Callee,
                            const Decl *DeclWithIssue) const {
    if (safeGetName(Callee) != "CFBridgingRelease" || CE->getNumArgs() != 1)
      return;

    auto *Arg = CE->getArg(0)->IgnoreParenCasts();
    auto *InnerCE = dyn_cast<CallExpr>(Arg);
    if (!InnerCE)
      return;

    auto *InnerF = InnerCE->getDirectCallee();
    if (!InnerF || !isCreateOrCopyFunction(InnerF))
      return;

    CreateOrCopyFnCall.insert(InnerCE);
  }

  void visitConstructExpr(const CXXConstructExpr *CE,
                          const Decl *DeclWithIssue) const {
    assert(BR && "expected nonnull BugReporter");
    if (BR->getSourceManager().isInSystemHeader(CE->getExprLoc()))
      return;

    auto *Ctor = CE->getConstructor();
    if (!Ctor)
      return;

    auto *Cls = Ctor->getParent();
    if (!Cls)
      return;

    if (!isRetainPtrOrOSPtr(safeGetName(Cls)) || !CE->getNumArgs())
      return;

    // Ignore RetainPtr construction inside adoptNS, adoptCF, and retainPtr.
    if (isAdoptFn(DeclWithIssue) || safeGetName(DeclWithIssue) == "retainPtr")
      return;

    std::string Name = "RetainPtr constructor";
    auto *Arg = CE->getArg(0)->IgnoreParenCasts();
    auto Result = isOwned(Arg);

    if (isCreateOrCopy(Arg))
      CreateOrCopyFnCall.insert(Arg); // Avoid double reporting.

    const Expr *Inner = nullptr;
    if (isAllocInit(Arg, &Inner)) {
      CreateOrCopyFnCall.insert(Arg);
      if (Inner)
        CreateOrCopyFnCall.insert(Inner);
    }

    if (Result == IsOwnedResult::Skip)
      return;

    if (Result == IsOwnedResult::Unknown)
      Result = IsOwnedResult::NotOwned;
    if (Result == IsOwnedResult::Owned)
      reportLeak(Name, CE, DeclWithIssue);
    else if (RTC.isARCEnabled() && isAllocInit(Arg))
      reportLeak(Name, CE, DeclWithIssue, "when ARC is disabled");
    else if (isCreateOrCopy(Arg))
      reportLeak(Name, CE, DeclWithIssue);
  }

  void visitVarDecl(const VarDecl *VD) const {
    auto *Init = VD->getInit();
    if (!Init || !RTC.isARCEnabled())
      return;
    Init = Init->IgnoreParenCasts();
    const Expr *Inner = nullptr;
    if (isAllocInit(Init, &Inner)) {
      CreateOrCopyFnCall.insert(Init);
      if (Inner)
        CreateOrCopyFnCall.insert(Inner);
    }
  }

  void visitBinaryOperator(const BinaryOperator *BO) const {
    if (!BO->isAssignmentOp())
      return;
    auto *LHS = BO->getLHS();
    auto *RHS = BO->getRHS()->IgnoreParenCasts();
    if (isa<ObjCIvarRefExpr>(LHS)) {
      const Expr *Inner = nullptr;
      if (isAllocInit(RHS, &Inner)) {
        CreateOrCopyFnCall.insert(RHS);
        if (Inner)
          CreateOrCopyFnCall.insert(Inner);
      }
      return;
    }
    auto *UO = dyn_cast<UnaryOperator>(LHS);
    if (!UO)
      return;
    auto OpCode = UO->getOpcode();
    if (OpCode != UO_Deref)
      return;
    auto *DerefTarget = UO->getSubExpr();
    if (!DerefTarget)
      return;
    DerefTarget = DerefTarget->IgnoreParenCasts();
    auto *DRE = dyn_cast<DeclRefExpr>(DerefTarget);
    if (!DRE)
      return;
    auto *Decl = DRE->getDecl();
    if (!Decl)
      return;
    if (!isa<ParmVarDecl>(Decl) || !isCreateOrCopy(RHS))
      return;
    if (Decl->hasAttr<CFReturnsRetainedAttr>())
      CreateOrCopyFnCall.insert(RHS);
  }

  void visitReturnStmt(const ReturnStmt *RS, const Decl *DeclWithIssue) const {
    if (!DeclWithIssue)
      return;
    auto *RawRetValue = RS->getRetValue();
    if (!RawRetValue)
      return;
    auto *RetValue = RawRetValue->IgnoreParenCasts();
    std::optional<bool> retainsRet;
    if (auto *FnDecl = dyn_cast<FunctionDecl>(DeclWithIssue))
      retainsRet = retainsReturnValue(FnDecl);
    else if (auto *MethodDecl = dyn_cast<ObjCMethodDecl>(DeclWithIssue))
      retainsRet = retainsReturnValue(MethodDecl);
    else
      return;
    if (retainsRet && *retainsRet) {
      // The caller of this function owns the returned value so any +1 value
      // which flows into the return value is not leaked here.
      markOwnershipTransferred(RetValue);
      // Conversely, returning a +0 value results in an over-release.
      checkReturnsPlusZero(RawRetValue, DeclWithIssue);
      return;
    }
    // Under ARC, returning [[X alloc] init] doesn't leak X.
    if (RTC.isUnretained(RetValue->getType()))
      return;
    if (auto *CE = dyn_cast<CallExpr>(RetValue)) {
      auto *Callee = CE->getDirectCallee();
      if (!Callee || !isCreateOrCopyFunction(Callee))
        return;
      CreateOrCopyFnCall.insert(CE);
      return;
    }
    const Expr *Inner = nullptr;
    if (isAllocInit(RetValue, &Inner)) {
      CreateOrCopyFnCall.insert(RetValue);
      if (Inner)
        CreateOrCopyFnCall.insert(Inner);
    }
  }

  /// Marks every expression the value of \p E may originate from as having its
  /// ownership transferred to the caller so that it's not reported as a leak.
  void markOwnershipTransferred(const Expr *E, unsigned Depth = 0) const {
    static constexpr unsigned MaxDepth = 16;
    if (!E || Depth > MaxDepth)
      return;
    E = E->IgnoreParenCasts();
    if (auto *BTE = dyn_cast<CXXBindTemporaryExpr>(E)) {
      CreateOrCopyFnCall.insert(E);
      markOwnershipTransferred(BTE->getSubExpr(), Depth + 1);
      return;
    }
    if (auto *POE = dyn_cast<PseudoObjectExpr>(E)) {
      if (unsigned SemanticExprCount = POE->getNumSemanticExprs()) {
        CreateOrCopyFnCall.insert(E);
        markOwnershipTransferred(POE->getSemanticExpr(SemanticExprCount - 1),
                                 Depth + 1);
        return;
      }
    }
    if (auto *CO = dyn_cast<AbstractConditionalOperator>(E)) {
      markOwnershipTransferred(CO->getTrueExpr(), Depth + 1);
      markOwnershipTransferred(CO->getFalseExpr(), Depth + 1);
      return;
    }
    if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      // A local variable holding onto the +1 value until it's returned.
      auto *VD = dyn_cast_or_null<VarDecl>(DRE->getDecl());
      if (VD && VD->hasLocalStorage() && !isa<ParmVarDecl>(VD))
        markOwnershipTransferred(VD->getInit(), Depth + 1);
      return;
    }
    CreateOrCopyFnCall.insert(E);
    const Expr *Inner = nullptr;
    if (isAllocInit(E, &Inner)) {
      if (Inner)
        CreateOrCopyFnCall.insert(Inner);
      return;
    }
    if (auto *CE = dyn_cast<CallExpr>(E)) {
      // Casting and bridging helpers simply pass the ownership of their
      // argument through to their return value.
      if (isOwnershipPassThroughFunction(CE))
        markOwnershipTransferred(CE->getArg(0), Depth + 1);
    }
  }

  bool isOwnershipPassThroughFunction(const CallExpr *CE) const {
    auto *Callee = CE->getDirectCallee();
    if (!Callee || CE->getNumArgs() != 1)
      return false;
    auto Name = safeGetName(Callee);
    return Name == "bridge_cast" || Name == "bridge_id_cast" ||
           Name == "checked_cf_cast" || Name == "dynamic_cf_cast" ||
           Name == "checked_objc_cast" || Name == "dynamic_objc_cast";
  }

  /// Reports returning a +0 value from a function which is expected to return
  /// a +1 value, which results in an over-release by the caller.
  void checkReturnsPlusZero(const Expr *RawRetValue,
                            const Decl *DeclWithIssue) const {
    // A __bridge_retained cast produces a +1 value out of a +0 one but
    // IgnoreParenCasts would strip it away.
    for (auto *E = RawRetValue; E;) {
      E = E->IgnoreParens();
      if (isa<ObjCBridgedCastExpr>(E))
        return;
      auto *Cast = dyn_cast<CastExpr>(E);
      E = Cast ? Cast->getSubExpr() : nullptr;
    }
    checkReturnValueIsOwned(RawRetValue->IgnoreParenCasts(), DeclWithIssue);
  }

  void checkReturnValueIsOwned(const Expr *RetValue,
                               const Decl *DeclWithIssue) const {
    if (auto *CO = dyn_cast<AbstractConditionalOperator>(RetValue)) {
      checkReturnValueIsOwned(CO->getTrueExpr()->IgnoreParenCasts(),
                              DeclWithIssue);
      checkReturnValueIsOwned(CO->getFalseExpr()->IgnoreParenCasts(),
                              DeclWithIssue);
      return;
    }
    // Returning null is always allowed.
    if (isNullPtr(RetValue))
      return;
    if (auto *DRE = dyn_cast<DeclRefExpr>(RetValue)) {
      auto *VD = dyn_cast_or_null<VarDecl>(DRE->getDecl());
      // A local variable may be assigned a +1 value after its declaration so
      // don't infer the retain count of the returned value from it.
      if (VD && VD->hasLocalStorage() && !isa<ParmVarDecl>(VD))
        return;
      // The ownership of a consumed parameter belongs to this function.
      if (VD &&
          (VD->hasAttr<CFConsumedAttr>() || VD->hasAttr<NSConsumedAttr>()))
        return;
    }
    // Only raw CF / NS pointers can be returned at the wrong retain count.
    // Under ARC, the compiler retains Objective-C objects as needed.
    if (!RTC.isUnretained(RetValue->getType()))
      return;
    if (isOwned(RetValue) != IsOwnedResult::NotOwned)
      return;
    reportUseAfterFreeForReturn(RetValue, DeclWithIssue);
  }

  template <typename CallableType>
  std::optional<bool> retainsReturnValue(const CallableType *FnDecl) const {
    auto Summary = Summaries->getSummary(AnyCall(FnDecl));
    auto RetEffect = Summary->getRetEffect();
    switch (RetEffect.getKind()) {
    case RetEffect::NoRet:
      return std::nullopt;
    case RetEffect::OwnedSymbol:
      return true;
    case RetEffect::NotOwnedSymbol:
      return false;
    case RetEffect::OwnedWhenTrackedReceiver:
      return std::nullopt;
    case RetEffect::NoRetHard:
      return std::nullopt;
    }
    return std::nullopt;
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
      if (auto *POE = dyn_cast<PseudoObjectExpr>(E)) {
        if (unsigned SemanticExprCount = POE->getNumSemanticExprs()) {
          E = POE->getSemanticExpr(SemanticExprCount - 1);
          continue;
        }
      }
      if (isNullPtr(E))
        return IsOwnedResult::NotOwned;
      if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
        auto QT = DRE->getType();
        if (isRetainPtrOrOSPtrType(QT))
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
            if (isRetainPtrOrOSPtr(safeGetName(Cls)) && ResultType &&
                (ResultType->isPointerType() || ResultType->isReferenceType() ||
                 ResultType->isObjCObjectPointerType()))
              return IsOwnedResult::NotOwned;
          }
          if (safeGetName(MD) == "leakRef" &&
              isRetainPtrOrOSPtr(safeGetName(Cls)))
            return IsOwnedResult::Owned;
          if (safeGetName(MD) == "get" && isRetainPtrOrOSPtr(safeGetName(Cls)))
            return IsOwnedResult::NotOwned;
        }
      }
      if (auto *CE = dyn_cast<CallExpr>(E)) {
        if (auto *Callee = CE->getDirectCallee()) {
          if (isAdoptFn(Callee))
            return IsOwnedResult::NotOwned;
          auto Name = safeGetName(Callee);
          if (Name == "__builtin___CFStringMakeConstantString")
            return IsOwnedResult::NotOwned;
          if ((Name == "checked_cf_cast" || Name == "dynamic_cf_cast" ||
               Name == "checked_objc_cast" || Name == "dynamic_objc_cast") &&
              CE->getNumArgs() == 1) {
            E = CE->getArg(0)->IgnoreParenCasts();
            continue;
          }
          auto RetType = Callee->getReturnType();
          if (isRetainPtrOrOSPtrType(RetType))
            return IsOwnedResult::NotOwned;
          if (isCreateOrCopyFunction(Callee)) {
            CreateOrCopyFnCall.insert(CE);
            return IsOwnedResult::Owned;
          }
        } else if (auto *CalleeExpr = CE->getCallee()) {
          if (isa<CXXDependentScopeMemberExpr>(CalleeExpr))
            return IsOwnedResult::Skip; // Wait for instantiation.
          if (isa<UnresolvedLookupExpr>(CalleeExpr))
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

  void reportUseAfterFree(const std::string &Name, const CallExpr *CE,
                          const Decl *DeclWithIssue,
                          const char *condition = nullptr) const {
    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    Os << "Incorrect use of " << Name
       << ". The argument is +0 and results in an use-after-free";
    if (condition)
      Os << " " << condition;
    Os << ".";

    assert(BR && "expected nonnull BugReporter");
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

    assert(BR && "expected nonnull BugReporter");
    PathDiagnosticLocation BSLoc(CE->getSourceRange().getBegin(),
                                 BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(CE->getSourceRange());
    Report->setDeclWithIssue(DeclWithIssue);
    BR->emitReport(std::move(Report));
  }

  template <typename ExprType>
  void reportLeak(const ExprType *E, const Decl *DeclWithIssue) const {
    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    Os << "The return value is +1 and results in a memory leak.";

    PathDiagnosticLocation BSLoc(E->getSourceRange().getBegin(),
                                 BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(E->getSourceRange());
    Report->setDeclWithIssue(DeclWithIssue);
    BR->emitReport(std::move(Report));
  }

  void reportUseAfterFreeForReturn(const Expr *E,
                                   const Decl *DeclWithIssue) const {
    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    Os << "The function is expected to return +1 but the return value is +0, "
          "which results in an use-after-free.";

    assert(BR && "expected nonnull BugReporter");
    PathDiagnosticLocation BSLoc(E->getSourceRange().getBegin(),
                                 BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(E->getSourceRange());
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
