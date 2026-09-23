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
#include "RawPtrRefSafetyModel.h"
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
  bool GuardianIsRawPtrOrRef{false};

  explicit GuardianVisitor(const VarDecl *Guardian,
                           bool GuardianIsRawPtrOrRef = false)
      : Guardian(Guardian), GuardianIsRawPtrOrRef(GuardianIsRawPtrOrRef) {
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
    auto *Ctor = CE->getConstructor();
    if (!Ctor)
      return false;
    unsigned ArgIndex = 0;
    for (auto *Arg : CE->arguments()) {
      ParmVarDecl *Parm = nullptr;
      if (ArgIndex < Ctor->getNumParams())
        Parm = Ctor->getParamDecl(ArgIndex);
      if (mutatesGuardian(Arg, Parm))
        return false;
      ArgIndex++;
    }
    return true;
  }

  bool VisitCallExpr(CallExpr *CE) override {
    auto *Callee = CE->getDirectCallee();
    if (!Callee)
      return false;
    if (isPtrConversion(Callee))
      return true;
    if (auto *Method = dyn_cast<CXXMethodDecl>(Callee)) {
      if (isGetterOfSafePtr(Method).value_or(false))
        return true;
    }
    unsigned ArgIndex = 0;
    unsigned ArgOffset = isa<CXXOperatorCallExpr>(CE);
    for (auto *Arg : CE->arguments()) {
      ParmVarDecl *Parm = nullptr;
      if (ArgIndex >= ArgOffset) {
        unsigned ParmIndex = ArgIndex - ArgOffset;
        if (ParmIndex < Callee->getNumParams())
          Parm = Callee->getParamDecl(ParmIndex);
      }
      if (mutatesGuardian(Arg, Parm))
        return false;
      ArgIndex++;
    }
    return true;
  }

  bool VisitCXXMemberCallExpr(CXXMemberCallExpr *MCE) override {
    if (GuardianIsRawPtrOrRef)
      return true;
    auto *Method = MCE->getMethodDecl();
    auto ObjType = MCE->getObjectType();
    if (ObjType.isConstQualified())
      return true;
    auto *ThisArg = MCE->getImplicitObjectArgument()->IgnoreParenCasts();
    if (auto *VarRef = dyn_cast<DeclRefExpr>(ThisArg)) {
      if (!isa<CXXConversionDecl>(Method) && VarRef->getDecl() == Guardian)
        return false;
    }
    return true;
  }

private:
  bool mutatesGuardian(const Expr *Arg, const ParmVarDecl *ParmDecl) {
    Arg = Arg->IgnoreParenCasts();
    auto ArgType = ParmDecl ? ParmDecl->getType() : Arg->getType();
    bool IsAddressOf = false;
    if (auto *UO = dyn_cast<UnaryOperator>(Arg);
        UO && UO->getOpcode() == UO_AddrOf) {
      Arg = UO->getSubExpr()->IgnoreParenCasts();
      IsAddressOf = true;
    }
    auto *VarRef = dyn_cast<DeclRefExpr>(Arg);
    if (!VarRef || VarRef->getDecl() != Guardian)
      return false;
    if (GuardianIsRawPtrOrRef && !Guardian->getType()->isPointerType())
      return false;
    if (IsAddressOf) {
      if (!ArgType->isPointerType())
        return false;
      return !ArgType->getPointeeType().isConstQualified();
    }
    if (GuardianIsRawPtrOrRef) {
      if (!ArgType->isReferenceType())
        return false;
      return !ArgType.getNonReferenceType().isConstQualified();
    }
    return !ArgType.isConstQualified();
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

static const VarDecl *findAssignedVar(const Expr *DestExpr) {
  while (DestExpr) {
    DestExpr = DestExpr->IgnoreParenCasts();
    if (auto *DRE = dyn_cast<DeclRefExpr>(DestExpr))
      return dyn_cast_or_null<VarDecl>(DRE->getDecl());
    if (auto *UO = dyn_cast<UnaryOperator>(DestExpr);
        UO && UO->getOpcode() == UO_Deref) {
      DestExpr = UO->getSubExpr();
      continue;
    }
    if (auto *ASE = dyn_cast<ArraySubscriptExpr>(DestExpr)) {
      DestExpr = ASE->getBase();
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

class RawPtrRefLocalVarsChecker
    : public Checker<check::ASTDecl<TranslationUnitDecl>> {
  BugType Bug;
  EnsureFunctionAnalysis EFA;

protected:
  mutable BugReporter *BR;
  const std::unique_ptr<PtrRefSafetyModel> Model;

public:
  RawPtrRefLocalVarsChecker(const char *description,
                            std::unique_ptr<PtrRefSafetyModel> Model)
      : Bug(this, description, "WebKit coding guidelines"),
        Model(std::move(Model)) {}

  std::optional<bool> isUnsafePtr(QualType T) const {
    return isUnsafePtrForStorage(*Model, T);
  }

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
        if (auto *RTC = Checker->Model->retainTypeChecker())
          RTC->visitTypedef(TD);
        return true;
      }

      bool VisitVarDecl(VarDecl *V) override {
        auto *Init = V->getInit();
        if (V->isLocalVarDecl())
          Checker->visitVarDecl(V, V->getType(), Init, DeclWithIssue);
        return true;
      }

      bool VisitBinaryOperator(BinaryOperator *BO) override {
        if (BO->isAssignmentOp()) {
          if (Checker->Model->recognizesIndirectStores()) {
            if (auto *V = findAssignedVar(BO->getLHS()))
              Checker->visitVarDecl(V, BO->getLHS()->getType(), BO->getRHS(),
                                    DeclWithIssue);
          } else if (auto *VarRef = dyn_cast<DeclRefExpr>(BO->getLHS())) {
            if (auto *V = dyn_cast<VarDecl>(VarRef->getDecl()))
              Checker->visitVarDecl(V, V->getType(), BO->getRHS(),
                                    DeclWithIssue);
          }
        }
        return true;
      }

      bool TraverseIfStmt(IfStmt *IS) override {
        if (IS->getConditionVariable()) {
          // This code currently does not explicitly check the "else" statement
          // since getConditionVariable returns nullptr when there is a
          // condition defined after ";" as in "if (auto foo = ~; !foo)". If
          // this semantics change, we should add an explicit check for "else".
          if (auto *Then = IS->getThen(); !Then || TFA.isTrivial(Then))
            return true;
        }
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
    if (auto *RTC = Model->retainTypeChecker())
      RTC->visitTranslationUnitDecl(TUD);
    visitor.TraverseDecl(const_cast<TranslationUnitDecl *>(TUD));
  }

  void visitVarDecl(const VarDecl *V, QualType SinkType, const Expr *Value,
                    const Decl *DeclWithIssue) const {
    if (shouldSkipVarDecl(V))
      return;

    if (auto *DD = dyn_cast<DecompositionDecl>(V)) {
      const auto *InitList =
          Value ? dyn_cast<InitListExpr>(Value->IgnoreParenCasts()) : nullptr;
      if (InitList && InitList->getNumInits() != DD->bindings().size())
        InitList = nullptr;

      unsigned BindingIndex = 0;
      for (auto *BD : DD->bindings()) {
        const unsigned Index = BindingIndex++;
        auto *Binding = BD->getBinding();
        if (!Binding)
          continue;
        std::optional<bool> IsUncountedPtr = isUnsafePtr(Binding->getType());
        if (!IsUncountedPtr || !*IsUncountedPtr)
          continue;

        const Expr *Origin = nullptr;
        if (Model->checksForInteriorDestruction()) {
          const Expr *Source = InitList ? InitList->getInit(Index) : Value;
          if (isPtrOriginSafe(V, Source, DeclWithIssue, Origin))
            continue;
        }
        reportBug(V, V->getType(), nullptr, BD, DeclWithIssue, Origin);
      }
    }

    std::optional<bool> IsUncountedPtr = isUnsafePtr(SinkType);
    if (IsUncountedPtr && *IsUncountedPtr) {
      const Expr *Origin = nullptr;
      if (Value) {
        if (isPtrOriginSafe(V, Value, DeclWithIssue, Origin))
          return;
      } else if (Model->checksForInteriorDestruction())
        return;
      reportBug(V, SinkType, Value, nullptr, DeclWithIssue, Origin);
    }
  }

  bool isPtrOriginSafe(const VarDecl *V, const Expr *Value,
                       const Decl *DeclWithIssue, const Expr *&Origin) const {
    return tryToFindPtrOrigin(
        Value, /*StopAtFirstRefCountedObj=*/false,
        Model->checksForInteriorDestruction(),
        [&](const clang::CXXRecordDecl *Record) {
          return Model->isSafePtr(Record);
        },
        [&](const clang::QualType Type) { return Model->isSafePtrType(Type); },
        [&](const clang::Decl *D) {
          return Model->isSafeDecl(D, BR->getSourceManager());
        },
        [&](const clang::Expr *InitArgOrigin, bool IsSafe,
            bool OriginDependsOnFullExpressionTemporary,
            bool PtrIsLifetimeBoundToOrigin) {
          if (!InitArgOrigin)
            return true;

          if (IsSafe) {
            if (!OriginDependsOnFullExpressionTemporary)
              return true;
            if (!Origin)
              Origin = InitArgOrigin;
            return false;
          }

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

          if (Model->isSafeExpr(InitArgOrigin, PtrIsLifetimeBoundToOrigin))
            return true;

          if (!Model->checksForInteriorDestruction() &&
              hasGuardian(V, InitArgOrigin, DeclWithIssue))
            return true;

          if (!Origin)
            Origin = InitArgOrigin;
          return false;
        });
  }

  bool hasGuardian(const VarDecl *V, const Expr *InitArgOrigin,
                   const Decl *DeclWithIssue) const {
    auto *Ref = dyn_cast<DeclRefExpr>(InitArgOrigin);
    if (!Ref)
      return false;

    auto *MaybeGuardian = dyn_cast_or_null<VarDecl>(Ref->getFoundDecl());
    if (!MaybeGuardian)
      return false;

    QualType GuardianType = MaybeGuardian->getType();
    if (!GuardianType.isNull()) {
      if (auto *Record = GuardianType->getAsCXXRecordDecl()) {
        if (MaybeGuardian->isLocalVarDecl() &&
            (Model->isSafePtr(Record) ||
             isRefcountedStringsHack(MaybeGuardian)) &&
            isGuardedScopeEmbeddedInGuardianScope(V, MaybeGuardian))
          return true;
      }
    }

    if (isa<ParmVarDecl>(MaybeGuardian)) {
      bool IsRawPtrOrRef = isUnsafePtr(GuardianType).value_or(false);
      GuardianVisitor Visitor{MaybeGuardian, IsRawPtrOrRef};
      if (auto *FD = dyn_cast<FunctionDecl>(DeclWithIssue))
        return Visitor.TraverseStmt(FD->getBody());
      if (auto *MD = dyn_cast<ObjCMethodDecl>(DeclWithIssue))
        return Visitor.TraverseStmt(MD->getBody());
    }

    return false;
  }

  bool shouldSkipVarDecl(const VarDecl *V) const {
    assert(V);
    if (isa<ImplicitParamDecl>(V))
      return true;
    if (V->isInitCapture())
      return true;
    return BR->getSourceManager().isInSystemHeader(V->getLocation());
  }

  void reportBug(const VarDecl *V, QualType SinkType, const Expr *Value,
                 const Decl *BindingDecl, const Decl *DeclWithIssue,
                 const Expr *Origin) const {
    assert(V);
    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    if (isa<ParmVarDecl>(V)) {
      Os << "Parameter ";
      printQuotedQualifiedName(Os, V);
      Os << " is a ";
      Model->describeHazard(Os, Origin, SinkType);

      SourceLocation ExprLoc = (Value) ? Value->getExprLoc() : V->getLocation();
      PathDiagnosticLocation BSLoc(ExprLoc, BR->getSourceManager());
      auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
      if (Value)
        Report->addRange(Value->getSourceRange());
      Report->setDeclWithIssue(DeclWithIssue);
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
      if (BindingDecl)
        Os << "'" << safeGetName(BindingDecl) << "'";
      else
        printQuotedQualifiedName(Os, V);
      Os << " is a ";
      Model->describeHazard(Os, Origin, SinkType);

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
                                  "provably backed by ref-counted variable",
                                  makeRefPtrSafetyModel()) {}
};

class UncheckedLocalVarsChecker final : public RawPtrRefLocalVarsChecker {
public:
  UncheckedLocalVarsChecker()
      : RawPtrRefLocalVarsChecker("Unchecked raw pointer or reference not "
                                  "provably backed by checked variable",
                                  makeCheckedPtrSafetyModel()) {}
};

class UnretainedLocalVarsChecker final : public RawPtrRefLocalVarsChecker {
public:
  UnretainedLocalVarsChecker()
      : RawPtrRefLocalVarsChecker("Unretained raw pointer or reference not "
                                  "provably backed by a RetainPtr",
                                  makeRetainPtrSafetyModel()) {}
};

class UnborrowedLocalVarsChecker final : public RawPtrRefLocalVarsChecker {
public:
  UnborrowedLocalVarsChecker()
      : RawPtrRefLocalVarsChecker("Loan on a CanBorrow object not guarded by "
                                  "a Borrow",
                                  makeBorrowSafetyModel()) {}
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

void ento::registerUnborrowedLocalVarsChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<UnborrowedLocalVarsChecker>();
}

bool ento::shouldRegisterUnborrowedLocalVarsChecker(const CheckerManager &) {
  return true;
}
