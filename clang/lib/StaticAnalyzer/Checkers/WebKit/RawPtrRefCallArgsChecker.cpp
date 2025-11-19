//=======- RawPtrRefCallArgsChecker.cpp --------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTUtils.h"
#include "DiagOutputUtils.h"
#include "PtrTypesSemantics.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/Analysis/DomainSpecific/CocoaConventions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "llvm/Support/SaveAndRestore.h"
#include <optional>

using namespace clang;
using namespace ento;

namespace {

class RawPtrRefCallArgsChecker
    : public Checker<check::ASTDecl<TranslationUnitDecl>> {
  BugType Bug;

  TrivialFunctionAnalysis TFA;
  EnsureFunctionAnalysis EFA;

protected:
  mutable BugReporter *BR;
  mutable std::optional<RetainTypeChecker> RTC;

public:
  RawPtrRefCallArgsChecker(const char *description)
      : Bug(this, description, "WebKit coding guidelines") {}

  virtual std::optional<bool> isUnsafeType(QualType) const = 0;
  virtual std::optional<bool> isUnsafePtr(QualType) const = 0;
  virtual bool isSafePtr(const CXXRecordDecl *Record) const = 0;
  virtual bool isSafePtrType(const QualType type) const = 0;
  virtual bool isSafeExpr(const Expr *) const { return false; }
  virtual bool isSafeDecl(const Decl *) const { return false; }
  virtual const char *ptrKind() const = 0;

  void checkASTDecl(const TranslationUnitDecl *TUD, AnalysisManager &MGR,
                    BugReporter &BRArg) const {
    BR = &BRArg;

    // The calls to checkAST* from AnalysisConsumer don't
    // visit template instantiations or lambda classes. We
    // want to visit those, so we make our own RecursiveASTVisitor.
    struct LocalVisitor : DynamicRecursiveASTVisitor {
      const RawPtrRefCallArgsChecker *Checker;
      Decl *DeclWithIssue{nullptr};

      explicit LocalVisitor(const RawPtrRefCallArgsChecker *Checker)
          : Checker(Checker) {
        assert(Checker);
        ShouldVisitTemplateInstantiations = true;
        ShouldVisitImplicitCode = false;
      }

      bool TraverseClassTemplateDecl(ClassTemplateDecl *Decl) override {
        if (isSmartPtrClass(safeGetName(Decl)))
          return true;
        return DynamicRecursiveASTVisitor::TraverseClassTemplateDecl(Decl);
      }

      bool TraverseDecl(Decl *D) override {
        llvm::SaveAndRestore SavedDecl(DeclWithIssue);
        if (D && (isa<FunctionDecl>(D) || isa<ObjCMethodDecl>(D)))
          DeclWithIssue = D;
        return DynamicRecursiveASTVisitor::TraverseDecl(D);
      }

      bool VisitCallExpr(CallExpr *CE) override {
        Checker->visitCallExpr(CE, DeclWithIssue);
        return true;
      }

      bool VisitTypedefDecl(TypedefDecl *TD) override {
        if (Checker->RTC)
          Checker->RTC->visitTypedef(TD);
        return true;
      }

      bool VisitObjCMessageExpr(ObjCMessageExpr *ObjCMsgExpr) override {
        Checker->visitObjCMessageExpr(ObjCMsgExpr, DeclWithIssue);
        return true;
      }
    };

    LocalVisitor visitor(this);
    if (RTC)
      RTC->visitTranslationUnitDecl(TUD);
    visitor.TraverseDecl(const_cast<TranslationUnitDecl *>(TUD));
  }

  void visitCallExpr(const CallExpr *CE, const Decl *D) const {
    if (shouldSkipCall(CE))
      return;

    if (auto *F = CE->getDirectCallee()) {
      // Skip the first argument for overloaded member operators (e. g. lambda
      // or std::function call operator).
      unsigned ArgIdx =
          isa<CXXOperatorCallExpr>(CE) && isa_and_nonnull<CXXMethodDecl>(F);

      if (auto *MemberCallExpr = dyn_cast<CXXMemberCallExpr>(CE)) {
        if (auto *MD = MemberCallExpr->getMethodDecl()) {
          auto name = safeGetName(MD);
          if (name == "ref" || name == "deref")
            return;
          if (name == "incrementCheckedPtrCount" ||
              name == "decrementCheckedPtrCount")
            return;
        }
        auto *E = MemberCallExpr->getImplicitObjectArgument();
        QualType ArgType = MemberCallExpr->getObjectType().getCanonicalType();
        std::optional<bool> IsUnsafe = isUnsafeType(ArgType);
        if (IsUnsafe && *IsUnsafe && !isPtrOriginSafe(E))
          reportBugOnThis(E, D);
      }

      for (auto P = F->param_begin();
           // FIXME: Also check variadic function parameters.
           // FIXME: Also check default function arguments. Probably a different
           // checker. In case there are default arguments the call can have
           // fewer arguments than the callee has parameters.
           P < F->param_end() && ArgIdx < CE->getNumArgs(); ++P, ++ArgIdx) {
        // TODO: attributes.
        // if ((*P)->hasAttr<SafeRefCntblRawPtrAttr>())
        //  continue;

        QualType ArgType = (*P)->getType();
        // FIXME: more complex types (arrays, references to raw pointers, etc)
        std::optional<bool> IsUncounted = isUnsafePtr(ArgType);
        if (!IsUncounted || !(*IsUncounted))
          continue;

        const auto *Arg = CE->getArg(ArgIdx);

        if (auto *defaultArg = dyn_cast<CXXDefaultArgExpr>(Arg))
          Arg = defaultArg->getExpr();

        if (isPtrOriginSafe(Arg))
          continue;

        reportBug(Arg, *P, D);
      }
      for (; ArgIdx < CE->getNumArgs(); ++ArgIdx) {
        const auto *Arg = CE->getArg(ArgIdx);
        auto ArgType = Arg->getType();
        std::optional<bool> IsUncounted = isUnsafePtr(ArgType);
        if (!IsUncounted || !(*IsUncounted))
          continue;

        if (auto *defaultArg = dyn_cast<CXXDefaultArgExpr>(Arg))
          Arg = defaultArg->getExpr();

        if (isPtrOriginSafe(Arg))
          continue;

        reportBug(Arg, nullptr, D);
      }
    }
  }

  void visitObjCMessageExpr(const ObjCMessageExpr *E, const Decl *D) const {
    if (BR->getSourceManager().isInSystemHeader(E->getExprLoc()))
      return;

    auto Selector = E->getSelector();
    if (auto *Receiver = E->getInstanceReceiver()) {
      std::optional<bool> IsUnsafe = isUnsafePtr(E->getReceiverType());
      if (IsUnsafe && *IsUnsafe && !isPtrOriginSafe(Receiver)) {
        if (auto *InnerMsg = dyn_cast<ObjCMessageExpr>(Receiver)) {
          auto InnerSelector = InnerMsg->getSelector();
          if (InnerSelector.getNameForSlot(0) == "alloc" &&
              Selector.getNameForSlot(0).starts_with("init"))
            return;
        }
        reportBugOnReceiver(Receiver, D);
      }
    }

    auto *MethodDecl = E->getMethodDecl();
    if (!MethodDecl)
      return;

    auto ArgCount = E->getNumArgs();
    for (unsigned i = 0; i < ArgCount; ++i) {
      auto *Arg = E->getArg(i);
      bool hasParam = i < MethodDecl->param_size();
      auto *Param = hasParam ? MethodDecl->getParamDecl(i) : nullptr;
      auto ArgType = Arg->getType();
      std::optional<bool> IsUnsafe = isUnsafePtr(ArgType);
      if (!IsUnsafe || !(*IsUnsafe))
        continue;
      if (isPtrOriginSafe(Arg))
        continue;
      reportBug(Arg, Param, D);
    }
  }

  bool isPtrOriginSafe(const Expr *Arg) const {
    return tryToFindPtrOrigin(
        Arg, /*StopAtFirstRefCountedObj=*/true,
        [&](const clang::CXXRecordDecl *Record) { return isSafePtr(Record); },
        [&](const clang::QualType T) { return isSafePtrType(T); },
        [&](const clang::Decl *D) { return isSafeDecl(D); },
        [&](const clang::Expr *ArgOrigin, bool IsSafe) {
          if (IsSafe)
            return true;
          if (isNullPtr(ArgOrigin))
            return true;
          if (isa<IntegerLiteral>(ArgOrigin)) {
            // FIXME: Check the value.
            // foo(123)
            return true;
          }
          if (isa<CXXBoolLiteralExpr>(ArgOrigin))
            return true;
          if (isa<ObjCStringLiteral>(ArgOrigin))
            return true;
          if (isASafeCallArg(ArgOrigin))
            return true;
          if (EFA.isACallToEnsureFn(ArgOrigin))
            return true;
          if (isSafeExpr(ArgOrigin))
            return true;
          return false;
        });
  }

  bool shouldSkipCall(const CallExpr *CE) const {
    const auto *Callee = CE->getDirectCallee();

    if (BR->getSourceManager().isInSystemHeader(CE->getExprLoc()))
      return true;

    if (Callee && TFA.isTrivial(Callee) && !Callee->isVirtualAsWritten())
      return true;

    if (isTrivialBuiltinFunction(Callee))
      return true;

    if (CE->getNumArgs() == 0)
      return false;

    // If an assignment is problematic we should warn about the sole existence
    // of object on LHS.
    if (auto *MemberOp = dyn_cast<CXXOperatorCallExpr>(CE)) {
      // Note: assignemnt to built-in type isn't derived from CallExpr.
      if (MemberOp->getOperator() ==
          OO_Equal) { // Ignore assignment to Ref/RefPtr.
        auto *callee = MemberOp->getDirectCallee();
        if (auto *calleeDecl = dyn_cast<CXXMethodDecl>(callee)) {
          if (const CXXRecordDecl *classDecl = calleeDecl->getParent()) {
            if (isSafePtr(classDecl))
              return true;
          }
        }
      }
      if (MemberOp->isAssignmentOp())
        return false;
    }

    if (!Callee)
      return false;

    if (isMethodOnWTFContainerType(Callee))
      return true;

    auto overloadedOperatorType = Callee->getOverloadedOperator();
    if (overloadedOperatorType == OO_EqualEqual ||
        overloadedOperatorType == OO_ExclaimEqual ||
        overloadedOperatorType == OO_LessEqual ||
        overloadedOperatorType == OO_GreaterEqual ||
        overloadedOperatorType == OO_Spaceship ||
        overloadedOperatorType == OO_AmpAmp ||
        overloadedOperatorType == OO_PipePipe)
      return true;

    if (isCtorOfSafePtr(Callee) || isPtrConversion(Callee))
      return true;

    auto name = safeGetName(Callee);
    if (name == "adoptRef" || name == "getPtr" || name == "WeakPtr" ||
        name == "is" || name == "equal" || name == "hash" || name == "isType" ||
        // FIXME: Most/all of these should be implemented via attributes.
        name == "CFEqual" || name == "equalIgnoringASCIICase" ||
        name == "equalIgnoringASCIICaseCommon" ||
        name == "equalIgnoringNullity" || name == "toString")
      return true;

    return false;
  }

  bool isMethodOnWTFContainerType(const FunctionDecl *Decl) const {
    if (!isa<CXXMethodDecl>(Decl))
      return false;
    auto *ClassDecl = Decl->getParent();
    if (!ClassDecl || !isa<CXXRecordDecl>(ClassDecl))
      return false;

    auto *NsDecl = ClassDecl->getParent();
    if (!NsDecl || !isa<NamespaceDecl>(NsDecl))
      return false;

    auto MethodName = safeGetName(Decl);
    auto ClsNameStr = safeGetName(ClassDecl);
    StringRef ClsName = ClsNameStr; // FIXME: Make safeGetName return StringRef.
    auto NamespaceName = safeGetName(NsDecl);
    // FIXME: These should be implemented via attributes.
    return NamespaceName == "WTF" &&
           (MethodName == "find" || MethodName == "findIf" ||
            MethodName == "reverseFind" || MethodName == "reverseFindIf" ||
            MethodName == "findIgnoringASCIICase" || MethodName == "get" ||
            MethodName == "inlineGet" || MethodName == "contains" ||
            MethodName == "containsIf" ||
            MethodName == "containsIgnoringASCIICase" ||
            MethodName == "startsWith" || MethodName == "endsWith" ||
            MethodName == "startsWithIgnoringASCIICase" ||
            MethodName == "endsWithIgnoringASCIICase" ||
            MethodName == "substring") &&
           (ClsName.ends_with("Vector") || ClsName.ends_with("Set") ||
            ClsName.ends_with("Map") || ClsName == "StringImpl" ||
            ClsName.ends_with("String"));
  }

  void reportBug(const Expr *CallArg, const ParmVarDecl *Param,
                 const Decl *DeclWithIssue) const {
    assert(CallArg);

    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    const std::string paramName = safeGetName(Param);
    Os << "Call argument";
    if (!paramName.empty()) {
      Os << " for parameter ";
      printQuotedQualifiedName(Os, Param);
    }
    Os << " is " << ptrKind() << " and unsafe.";

    bool usesDefaultArgValue = isa<CXXDefaultArgExpr>(CallArg) && Param;
    const SourceLocation SrcLocToReport =
        usesDefaultArgValue ? Param->getDefaultArg()->getExprLoc()
                            : CallArg->getSourceRange().getBegin();

    PathDiagnosticLocation BSLoc(SrcLocToReport, BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(CallArg->getSourceRange());
    Report->setDeclWithIssue(DeclWithIssue);
    BR->emitReport(std::move(Report));
  }

  void reportBugOnThis(const Expr *CallArg, const Decl *DeclWithIssue) const {
    assert(CallArg);

    const SourceLocation SrcLocToReport = CallArg->getSourceRange().getBegin();

    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);
    Os << "Call argument for 'this' parameter is " << ptrKind();
    Os << " and unsafe.";

    PathDiagnosticLocation BSLoc(SrcLocToReport, BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(CallArg->getSourceRange());
    Report->setDeclWithIssue(DeclWithIssue);
    BR->emitReport(std::move(Report));
  }

  void reportBugOnReceiver(const Expr *CallArg,
                           const Decl *DeclWithIssue) const {
    assert(CallArg);

    const SourceLocation SrcLocToReport = CallArg->getSourceRange().getBegin();

    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);
    Os << "Receiver is " << ptrKind() << " and unsafe.";

    PathDiagnosticLocation BSLoc(SrcLocToReport, BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(CallArg->getSourceRange());
    Report->setDeclWithIssue(DeclWithIssue);
    BR->emitReport(std::move(Report));
  }
};

class UncountedCallArgsChecker final : public RawPtrRefCallArgsChecker {
public:
  UncountedCallArgsChecker()
      : RawPtrRefCallArgsChecker("Uncounted call argument for a raw "
                                 "pointer/reference parameter") {}

  std::optional<bool> isUnsafeType(QualType QT) const final {
    return isUncounted(QT);
  }

  std::optional<bool> isUnsafePtr(QualType QT) const final {
    return isUncountedPtr(QT.getCanonicalType());
  }

  bool isSafePtr(const CXXRecordDecl *Record) const final {
    return isRefCounted(Record) || isCheckedPtr(Record);
  }

  bool isSafePtrType(const QualType type) const final {
    return isRefOrCheckedPtrType(type);
  }

  const char *ptrKind() const final { return "uncounted"; }
};

class UncheckedCallArgsChecker final : public RawPtrRefCallArgsChecker {
public:
  UncheckedCallArgsChecker()
      : RawPtrRefCallArgsChecker("Unchecked call argument for a raw "
                                 "pointer/reference parameter") {}

  std::optional<bool> isUnsafeType(QualType QT) const final {
    return isUnchecked(QT);
  }

  std::optional<bool> isUnsafePtr(QualType QT) const final {
    return isUncheckedPtr(QT.getCanonicalType());
  }

  bool isSafePtr(const CXXRecordDecl *Record) const final {
    return isRefCounted(Record) || isCheckedPtr(Record);
  }

  bool isSafePtrType(const QualType type) const final {
    return isRefOrCheckedPtrType(type);
  }

  bool isSafeExpr(const Expr *E) const final {
    return isExprToGetCheckedPtrCapableMember(E);
  }

  const char *ptrKind() const final { return "unchecked"; }
};

class UnretainedCallArgsChecker final : public RawPtrRefCallArgsChecker {
public:
  UnretainedCallArgsChecker()
      : RawPtrRefCallArgsChecker("Unretained call argument for a raw "
                                 "pointer/reference parameter") {
    RTC = RetainTypeChecker();
  }

  std::optional<bool> isUnsafeType(QualType QT) const final {
    return RTC->isUnretained(QT);
  }

  std::optional<bool> isUnsafePtr(QualType QT) const final {
    return RTC->isUnretained(QT);
  }

  bool isSafePtr(const CXXRecordDecl *Record) const final {
    return isRetainPtrOrOSPtr(Record);
  }

  bool isSafePtrType(const QualType type) const final {
    return isRetainPtrOrOSPtrType(type);
  }

  bool isSafeExpr(const Expr *E) const final {
    return ento::cocoa::isCocoaObjectRef(E->getType()) &&
           isa<ObjCMessageExpr>(E);
  }

  bool isSafeDecl(const Decl *D) const final {
    // Treat NS/CF globals in system header as immortal.
    return BR->getSourceManager().isInSystemHeader(D->getLocation());
  }

  const char *ptrKind() const final { return "unretained"; }
};

} // namespace

void ento::registerUncountedCallArgsChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<UncountedCallArgsChecker>();
}

bool ento::shouldRegisterUncountedCallArgsChecker(const CheckerManager &) {
  return true;
}

void ento::registerUncheckedCallArgsChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<UncheckedCallArgsChecker>();
}

bool ento::shouldRegisterUncheckedCallArgsChecker(const CheckerManager &) {
  return true;
}

void ento::registerUnretainedCallArgsChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<UnretainedCallArgsChecker>();
}

bool ento::shouldRegisterUnretainedCallArgsChecker(const CheckerManager &) {
  return true;
}
