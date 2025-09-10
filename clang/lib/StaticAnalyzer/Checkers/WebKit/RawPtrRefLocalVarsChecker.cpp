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
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/Analysis/DomainSpecific/CocoaConventions.h"
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

struct GuardianVisitor : DynamicRecursiveASTVisitor {
  const VarDecl *Guardian{nullptr};

  explicit GuardianVisitor(const VarDecl *Guardian) : Guardian(Guardian) {
    assert(Guardian);
  }

  bool VisitBinaryOperator(BinaryOperator *BO) override {
    if (BO->isAssignmentOp()) {
      if (auto *VarRef = dyn_cast<DeclRefExpr>(BO->getLHS())) {
        if (VarRef->getDecl() == Guardian)
          return false;
      }
    }
    return true;
  }

  bool VisitCXXConstructExpr(CXXConstructExpr *CE) override {
    if (auto *Ctor = CE->getConstructor()) {
      if (Ctor->isMoveConstructor() && CE->getNumArgs() == 1) {
        auto *Arg = CE->getArg(0)->IgnoreParenCasts();
        if (auto *VarRef = dyn_cast<DeclRefExpr>(Arg)) {
          if (VarRef->getDecl() == Guardian)
            return false;
        }
      }
    }
    return true;
  }

  bool VisitCXXMemberCallExpr(CXXMemberCallExpr *MCE) override {
    auto MethodName = safeGetName(MCE->getMethodDecl());
    if (MethodName == "swap" || MethodName == "leakRef" ||
        MethodName == "releaseNonNull" || MethodName == "clear") {
      auto *ThisArg = MCE->getImplicitObjectArgument()->IgnoreParenCasts();
      if (auto *VarRef = dyn_cast<DeclRefExpr>(ThisArg)) {
        if (VarRef->getDecl() == Guardian)
          return false;
      }
    }
    return true;
  }

  bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *OCE) override {
    if (OCE->isAssignmentOp()) {
      assert(OCE->getNumArgs() == 2);
      auto *ThisArg = OCE->getArg(0)->IgnoreParenCasts();
      if (auto *VarRef = dyn_cast<DeclRefExpr>(ThisArg)) {
        if (VarRef->getDecl() == Guardian)
          return false;
      }
    }
    return true;
  }
};

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
  const CompoundStmt *FirstCompondStmt = nullptr;
  for (DynTypedNodeList guardedVarAncestors = ctx.getParents(*Guarded);
       !guardedVarAncestors.empty();
       guardedVarAncestors = ctx.getParents(
           *guardedVarAncestors
                .begin()) // FIXME - should we handle all of the parents?
  ) {
    for (auto &guardedVarAncestor : guardedVarAncestors) {
      if (auto *CStmtAncestor = guardedVarAncestor.get<CompoundStmt>()) {
        if (!FirstCompondStmt) {
          FirstCompondStmt = CStmtAncestor;
          continue;
        }
        if (CStmtAncestor == guardiansClosestCompStmtAncestor) {
          GuardianVisitor guardianVisitor(MaybeGuardian);
          auto *GuardedScope = const_cast<CompoundStmt *>(FirstCompondStmt);
          return guardianVisitor.TraverseCompoundStmt(GuardedScope);
        }
      }
    }
  }

  return false;
}

class RawPtrRefLocalVarsChecker
    : public Checker<check::ASTDecl<TranslationUnitDecl>> {
  BugType Bug;
  mutable BugReporter *BR;
  EnsureFunctionAnalysis EFA;

protected:
  mutable std::optional<RetainTypeChecker> RTC;

public:
  RawPtrRefLocalVarsChecker(const char *description)
      : Bug(this, description, "WebKit coding guidelines") {}

  virtual std::optional<bool> isUnsafePtr(const QualType T) const = 0;
  virtual bool isSafePtr(const CXXRecordDecl *) const = 0;
  virtual bool isSafePtrType(const QualType) const = 0;
  virtual bool isSafeExpr(const Expr *) const { return false; }
  virtual const char *ptrKind() const = 0;

  void checkASTDecl(const TranslationUnitDecl *TUD, AnalysisManager &MGR,
                    BugReporter &BRArg) const {
    BR = &BRArg;

    // The calls to checkAST* from AnalysisConsumer don't
    // visit template instantiations or lambda classes. We
    // want to visit those, so we make our own RecursiveASTVisitor.
    struct LocalVisitor : DynamicRecursiveASTVisitor {
      const RawPtrRefLocalVarsChecker *Checker;
      Decl *DeclWithIssue{nullptr};

      TrivialFunctionAnalysis TFA;

      explicit LocalVisitor(const RawPtrRefLocalVarsChecker *Checker)
          : Checker(Checker) {
        assert(Checker);
        ShouldVisitTemplateInstantiations = true;
        ShouldVisitImplicitCode = false;
      }

      bool TraverseDecl(Decl *D) override {
        llvm::SaveAndRestore SavedDecl(DeclWithIssue);
        if (D && (isa<FunctionDecl>(D) || isa<ObjCMethodDecl>(D)))
          DeclWithIssue = D;
        return DynamicRecursiveASTVisitor::TraverseDecl(D);
      }

      bool VisitTypedefDecl(TypedefDecl *TD) override {
        if (Checker->RTC)
          Checker->RTC->visitTypedef(TD);
        return true;
      }

      bool VisitVarDecl(VarDecl *V) override {
        auto *Init = V->getInit();
        if (Init && V->isLocalVarDecl())
          Checker->visitVarDecl(V, Init, DeclWithIssue);
        return true;
      }

      bool VisitBinaryOperator(BinaryOperator *BO) override {
        if (BO->isAssignmentOp()) {
          if (auto *VarRef = dyn_cast<DeclRefExpr>(BO->getLHS())) {
            if (auto *V = dyn_cast<VarDecl>(VarRef->getDecl()))
              Checker->visitVarDecl(V, BO->getRHS(), DeclWithIssue);
          }
        }
        return true;
      }

      bool TraverseIfStmt(IfStmt *IS) override {
        if (!TFA.isTrivial(IS))
          return DynamicRecursiveASTVisitor::TraverseIfStmt(IS);
        return true;
      }

      bool TraverseForStmt(ForStmt *FS) override {
        if (!TFA.isTrivial(FS))
          return DynamicRecursiveASTVisitor::TraverseForStmt(FS);
        return true;
      }

      bool TraverseCXXForRangeStmt(CXXForRangeStmt *FRS) override {
        if (!TFA.isTrivial(FRS))
          return DynamicRecursiveASTVisitor::TraverseCXXForRangeStmt(FRS);
        return true;
      }

      bool TraverseWhileStmt(WhileStmt *WS) override {
        if (!TFA.isTrivial(WS))
          return DynamicRecursiveASTVisitor::TraverseWhileStmt(WS);
        return true;
      }

      bool TraverseCompoundStmt(CompoundStmt *CS) override {
        if (!TFA.isTrivial(CS))
          return DynamicRecursiveASTVisitor::TraverseCompoundStmt(CS);
        return true;
      }

      bool TraverseClassTemplateDecl(ClassTemplateDecl *Decl) override {
        if (isSmartPtrClass(safeGetName(Decl)))
          return true;
        return DynamicRecursiveASTVisitor::TraverseClassTemplateDecl(Decl);
      }
    };

    LocalVisitor visitor(this);
    if (RTC)
      RTC->visitTranslationUnitDecl(TUD);
    visitor.TraverseDecl(const_cast<TranslationUnitDecl *>(TUD));
  }

  void visitVarDecl(const VarDecl *V, const Expr *Value,
                    const Decl *DeclWithIssue) const {
    if (shouldSkipVarDecl(V))
      return;

    std::optional<bool> IsUncountedPtr = isUnsafePtr(V->getType());
    if (IsUncountedPtr && *IsUncountedPtr) {
      if (tryToFindPtrOrigin(
              Value, /*StopAtFirstRefCountedObj=*/false,
              [&](const clang::CXXRecordDecl *Record) {
                return isSafePtr(Record);
              },
              [&](const clang::QualType Type) { return isSafePtrType(Type); },
              [&](const clang::Expr *InitArgOrigin, bool IsSafe) {
                if (!InitArgOrigin || IsSafe)
                  return true;

                if (isa<CXXThisExpr>(InitArgOrigin))
                  return true;

                if (isNullPtr(InitArgOrigin))
                  return true;

                if (isa<IntegerLiteral>(InitArgOrigin))
                  return true;

                if (isConstOwnerPtrMemberExpr(InitArgOrigin))
                  return true;

                if (EFA.isACallToEnsureFn(InitArgOrigin))
                  return true;

                if (isSafeExpr(InitArgOrigin))
                  return true;

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
                            (isSafePtr(MaybeGuardianArgCXXRecord) ||
                             isRefcountedStringsHack(MaybeGuardian)) &&
                            isGuardedScopeEmbeddedInGuardianScope(
                                V, MaybeGuardian))
                          return true;
                      }
                    }

                    // Parameters are guaranteed to be safe for the duration of
                    // the call by another checker.
                    if (isa<ParmVarDecl>(MaybeGuardian))
                      return true;
                  }
                }

                return false;
              }))
        return;

      reportBug(V, Value, DeclWithIssue);
    }
  }

  bool shouldSkipVarDecl(const VarDecl *V) const {
    assert(V);
    if (isa<ImplicitParamDecl>(V))
      return true;
    return BR->getSourceManager().isInSystemHeader(V->getLocation());
  }

  void reportBug(const VarDecl *V, const Expr *Value,
                 const Decl *DeclWithIssue) const {
    assert(V);
    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    if (isa<ParmVarDecl>(V)) {
      Os << "Assignment to an " << ptrKind() << " parameter ";
      printQuotedQualifiedName(Os, V);
      Os << " is unsafe.";

      PathDiagnosticLocation BSLoc(Value->getExprLoc(), BR->getSourceManager());
      auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
      Report->addRange(Value->getSourceRange());
      BR->emitReport(std::move(Report));
    } else {
      if (V->hasLocalStorage())
        Os << "Local variable ";
      else if (V->isStaticLocal())
        Os << "Static local variable ";
      else if (V->hasGlobalStorage())
        Os << "Global variable ";
      else
        Os << "Variable ";
      printQuotedQualifiedName(Os, V);
      Os << " is " << ptrKind() << " and unsafe.";

      PathDiagnosticLocation BSLoc(V->getLocation(), BR->getSourceManager());
      auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
      Report->addRange(V->getSourceRange());
      Report->setDeclWithIssue(DeclWithIssue);
      BR->emitReport(std::move(Report));
    }
  }
};

class UncountedLocalVarsChecker final : public RawPtrRefLocalVarsChecker {
public:
  UncountedLocalVarsChecker()
      : RawPtrRefLocalVarsChecker("Uncounted raw pointer or reference not "
                                  "provably backed by ref-counted variable") {}
  std::optional<bool> isUnsafePtr(const QualType T) const final {
    return isUncountedPtr(T);
  }
  bool isSafePtr(const CXXRecordDecl *Record) const final {
    return isRefCounted(Record) || isCheckedPtr(Record);
  }
  bool isSafePtrType(const QualType type) const final {
    return isRefOrCheckedPtrType(type);
  }
  const char *ptrKind() const final { return "uncounted"; }
};

class UncheckedLocalVarsChecker final : public RawPtrRefLocalVarsChecker {
public:
  UncheckedLocalVarsChecker()
      : RawPtrRefLocalVarsChecker("Unchecked raw pointer or reference not "
                                  "provably backed by checked variable") {}
  std::optional<bool> isUnsafePtr(const QualType T) const final {
    return isUncheckedPtr(T);
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

class UnretainedLocalVarsChecker final : public RawPtrRefLocalVarsChecker {
public:
  UnretainedLocalVarsChecker()
      : RawPtrRefLocalVarsChecker("Unretained raw pointer or reference not "
                                  "provably backed by a RetainPtr") {
    RTC = RetainTypeChecker();
  }
  std::optional<bool> isUnsafePtr(const QualType T) const final {
    return RTC->isUnretained(T);
  }
  bool isSafePtr(const CXXRecordDecl *Record) const final {
    return isRetainPtr(Record);
  }
  bool isSafePtrType(const QualType type) const final {
    return isRetainPtrType(type);
  }
  bool isSafeExpr(const Expr *E) const final {
    return ento::cocoa::isCocoaObjectRef(E->getType()) &&
           isa<ObjCMessageExpr>(E);
  }
  const char *ptrKind() const final { return "unretained"; }
};

} // namespace

void ento::registerUncountedLocalVarsChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<UncountedLocalVarsChecker>();
}

bool ento::shouldRegisterUncountedLocalVarsChecker(const CheckerManager &) {
  return true;
}

void ento::registerUncheckedLocalVarsChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<UncheckedLocalVarsChecker>();
}

bool ento::shouldRegisterUncheckedLocalVarsChecker(const CheckerManager &) {
  return true;
}

void ento::registerUnretainedLocalVarsChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<UnretainedLocalVarsChecker>();
}

bool ento::shouldRegisterUnretainedLocalVarsChecker(const CheckerManager &) {
  return true;
}
