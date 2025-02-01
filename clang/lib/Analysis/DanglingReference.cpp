#include "clang/Analysis/Analyses/DanglingReference.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <cassert>
#include <sstream>

namespace clang {
namespace {

template <typename T> static bool isRecordWithAttr(QualType Type) {
  auto *RD = Type->getAsCXXRecordDecl();
  if (!RD)
    return false;
  bool Result = RD->hasAttr<T>();

  if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RD))
    Result |= CTSD->getSpecializedTemplate()->getTemplatedDecl()->hasAttr<T>();
  if (RD->hasDefinition())
    for (auto B : RD->bases())
      Result |= isRecordWithAttr<T>(B.getType());
  return Result;
}

bool isOwner(QualType Q) { return isRecordWithAttr<OwnerAttr>(Q); }
bool isOwner(const Expr *E) { return isOwner(E->getType()); }
bool isOwner(const Decl *D) {
  return isa<ValueDecl>(D) &&
         isRecordWithAttr<OwnerAttr>(dyn_cast<ValueDecl>(D)->getType());
}
bool isPointer(QualType Q) {
  return Q->isPointerType() || isRecordWithAttr<PointerAttr>(Q);
}
bool isPointer(const Expr *E) { return isPointer(E->getType()); }
bool isPointer(const Decl *D) {
  auto *VD = dyn_cast<ValueDecl>(D);
  return VD && isPointer(VD->getType());
}

struct MemoryLoc {
  enum Storage {
    EMPTY,   // Pointer is null.
    STACK,   // Pointer points to something on stack.
    UNKNOWN, // Pointer points to an unknown entity.
  } Loc;
  // Details of stack location.
  const Decl *D = nullptr;
  const Expr *MTE = nullptr;

  bool IsEmpty() { return Loc == EMPTY; }
  bool IsOnStack() { return Loc == STACK; }
  bool IsUnkown() { return Loc == UNKNOWN; }

  const Decl *getDecl() { return D; }
  const Expr *getExpr() { return MTE; }

  static MemoryLoc Unknown() { return {UNKNOWN, nullptr, nullptr}; }
  static MemoryLoc Empty() { return {EMPTY, nullptr, nullptr}; }
  static MemoryLoc VarOnStack(const Decl *D) { return {STACK, D, nullptr}; }
  static MemoryLoc Temporary(const Expr *MTE) { return {STACK, nullptr, MTE}; }

  std::string str() {
    std::ostringstream os;
    switch (Loc) {
    case EMPTY:
      os << "Empty";
      break;
    case UNKNOWN:
      os << "Unknown";
      break;
    case STACK:
      os << "Stack";
      if (auto *VD = dyn_cast_or_null<VarDecl>(D))
        os << " \"" << VD->getName().str() << "\"";
      if (MTE)
        os << " (temporary)";
      break;
    }
    return os.str();
  }
};

bool implicitObjectParamIsLifetimeBound(const FunctionDecl *FD) {
  const TypeSourceInfo *TSI = FD->getTypeSourceInfo();
  if (!TSI)
    return false;
  AttributedTypeLoc ATL;
  for (TypeLoc TL = TSI->getTypeLoc();
       (ATL = TL.getAsAdjusted<AttributedTypeLoc>());
       TL = ATL.getModifiedLoc()) {
    if (ATL.getAttrAs<LifetimeBoundAttr>())
      return true;
  }
  return false;
}

class PointsToTracker : public ConstStmtVisitor<PointsToTracker> {
  const DeclContext &DC;

public:
  PointsToTracker(const DeclContext &DC) : DC(DC) {}
  void Handle(const Stmt *S) {
    if (auto *E = dyn_cast<Expr>(S);
        E && ExprPointsTo.find(E) != ExprPointsTo.end())
      return;
    Visit(S);
  }

  void MaybeInitaliseDecl(const Decl *D) {
    if (!D)
      return;
    auto *VD = dyn_cast<VarDecl>(D);
    if (!VD)
      return;
    // Initialise the pointer if we are seeing it for the first time.
    if (isPointer(VD)) {
      if (DeclPointsTo.find(D) != DeclPointsTo.end())
        return;
      // Pointer params are always unknown.
      if (isa<ParmVarDecl>(VD)) {
        UpdatePointer(VD, MemoryLoc::Unknown());
        return;
      }
      UpdatePointer(VD, ResolveExpr(VD->getInit()));
      return;
    }
    // Ignore non-stack variable.
    // TODO: Track reference variables.
    if (!DC.containsDecl(const_cast<Decl *>(D)) || !VD->hasLocalStorage() ||
        VD->getType()->isReferenceType())
      return;

    if (StackDecls.find(D) == StackDecls.end())
      AddToStack(VD);
  }

  // Merge above and below in VisitVarDecl !!
  void VisitDeclStmt(const DeclStmt *DS) {
    MaybeInitaliseDecl(DS->getSingleDecl());
  }

  void VisitDeclRefExpr(const DeclRefExpr *DRE) {
    SetExprPointer(DRE, DRE->getDecl());
  }

  void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *MTE) {
    // Ignore MTE of pointer types.
    if (isPointer(MTE)) {
      Handle(MTE->getSubExpr());
      SetExprPointer(MTE, MTE->getSubExpr());
      return;
    }
    // We have a temporary on stack.
    AddToStack(MTE);
  }

  void VisitImplicitCastExpr(const ImplicitCastExpr *E) {
    auto *SE = E->IgnoreImpCasts();
    Handle(SE);
    SetExprPointer(E, SE);
  }

  void VisitExprWithCleanups(const ExprWithCleanups *E) {
    SetExprPointer(E, E->getSubExpr());
  }

  void VisitCallExpr(const CallExpr *CE) {
    // Conversion from Owner to a Pointer.
    if (auto *ConversionFrom = CE->IgnoreConversionOperatorSingleStep();
        ConversionFrom != CE) {
      if (isOwner(ConversionFrom) && isPointer(CE)) {
        SetExprPointer(CE, ConversionFrom);
        return;
      }
    }
    // Lifetimebound function calls.
    auto *FD = dyn_cast_or_null<FunctionDecl>(CE->getCalleeDecl());
    if (!FD)
      return;
    // FIXME: This should only be done for GSL pointer args and not all args!
    QualType RetType = FD->getReturnType();
    if (RetType->isReferenceType() && !isOwner(RetType->getPointeeType()))
      return;
    Expr *ObjectArg = nullptr;
    if (auto *MCE = dyn_cast<CXXMemberCallExpr>(CE)) {
      ObjectArg = MCE->getImplicitObjectArgument();
      if (ObjectArg && implicitObjectParamIsLifetimeBound(FD)) {
        // TODO: Track more args. Not just the first one!
        SetExprPointer(CE, ObjectArg);
        return;
      }
    }
    for (unsigned I = 0; I < FD->getNumParams(); ++I) {
      const ParmVarDecl *PVD = FD->getParamDecl(I);
      if (CE->getArg(I) && PVD->hasAttr<LifetimeBoundAttr>()) {
        // TODO: Track more args. Not just the first one!
        SetExprPointer(CE, CE->getArg(I));
        return;
      }
    }
  }

  void VisitCXXConstructExpr(const CXXConstructExpr *CCE) {
    if (!isPointer(CCE))
      return;
    if (CCE->getNumArgs() == 1)
      if (isOwner(CCE->getArg(0)) || isPointer(CCE->getArg(0)))
        SetExprPointer(CCE, CCE->getArg(0));
  }

  void VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *OCE) {
    if (OCE->isAssignmentOp()) {
      assert(OCE->getNumArgs() == 2);
      Handle(OCE->getArg(0));
      Handle(OCE->getArg(1));
      HandleAssignment(OCE->getArg(0), OCE->getArg(1));
    }
  }

private:
  // Returns the Decl that is aliased by this expression.
  const Decl *DeclReferencedBy(const Expr *E) {
    if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      return DRE->getDecl();
    }
    return nullptr;
  }

  void HandleAssignment(const Expr *A, const Expr *B) {
    if (!isPointer(A))
      return;

    if (const Decl *PointerDecl = DeclReferencedBy(A);
        PointerDecl && isPointer(PointerDecl))
      UpdatePointer(PointerDecl, B);
  }

  // Update the contents of a Pointer.
  void UpdatePointer(const Decl *PointerD, MemoryLoc ML) {
    assert(isPointer(PointerD));
    DeclPointsTo[PointerD] = ML;
  }
  void UpdatePointer(const Decl *PointerD, const Expr *A) {
    UpdatePointer(PointerD, ResolveExpr(A));
  }

  void SetExprPointer(const Expr *E, MemoryLoc ML) {
    // llvm::errs() << "SetExprPointer : " << "\n";
    // E->dumpColor();
    // llvm::errs() << "ML : " << ML.str() << "\n";
    assert(ExprPointsTo.find(E) == ExprPointsTo.end());
    ExprPointsTo[E] = ML;
  }
  void SetExprPointer(const Expr *E, const Expr *PointeeE) {
    SetExprPointer(E, ResolveExpr(PointeeE));
  }
  void SetExprPointer(const Expr *E, const Decl *D) {
    SetExprPointer(E, ResolveDecl(D));
  }

public:
  MemoryLoc ResolveExpr(const Expr *E) {
    if (!E)
      return MemoryLoc::Empty();
    auto Res = ExprPointsTo.find(E);
    if (Res != ExprPointsTo.end()) {
      return Res->getSecond();
    }
    Handle(E);
    Res = ExprPointsTo.find(E);
    if (Res != ExprPointsTo.end()) {
      return Res->getSecond();
    }
    return ExprPointsTo[E] = MemoryLoc::Unknown();
  }

  // Returns the memory location pointed to by D. If D is a pointer-type,
  // returns the memory pointed to by the pointer.
  MemoryLoc ResolveDecl(const Decl *D) {
    // llvm::errs() << "Resolving decl : " << "\n";
    // D->dumpColor();
    MaybeInitaliseDecl(D);
    if (isPointer(D))
      return ResolvePointer(D);
    return ResolveNonPointer(D);
  }

  MemoryLoc ResolvePointer(const Decl *D) {
    auto *VD = dyn_cast<VarDecl>(D);
    // TODO: Handle other decls like field.
    if (!VD || !VD->hasLocalStorage())
      return MemoryLoc::Unknown();
    assert(isPointer(D));
    auto Res = DeclPointsTo.find(D);
    assert(Res != DeclPointsTo.end());
    return Res->getSecond();
  }

  MemoryLoc ResolveNonPointer(const Decl *D) {
    assert(!isPointer(D));
    // llvm::errs() << "ResolveNonPointer : " << "\n";
    // D->dumpColor();
    if (IsOnStack(D)) {
      // llvm::errs() << "Resolved NonPointer : " << "On stack\n";
      return MemoryLoc::VarOnStack(D);
    }
    return MemoryLoc::Unknown();
  }

  void AddToStack(const Decl *D) {
    // llvm::errs() << "AddToStack : " << "\n";
    // D->dumpColor();
    StackDecls.insert(D);
  }
  void AddToStack(const Expr *E) {
    assert(isa<MaterializeTemporaryExpr>(E));
    assert(!isPointer(E));
    StackExprs.insert(E);
    // Add a self edge.
    assert(ExprPointsTo.find(E) == ExprPointsTo.end());
    ExprPointsTo[E] = MemoryLoc::Temporary(E);
  }
  bool IsOnStack(const Decl *D) { return StackDecls.contains(D); }
  bool IsOnStack(const Expr *E) { return StackExprs.contains(E); }

  // ExpressionResolver &ExprResolver;
  // Map from an expression of View type to its pointee and Owner type to the
  // reference<TODO simplify>. This should follow single assignment because
  // Expr* cannot be reassigned in the program.
  llvm::DenseMap<const Expr *, MemoryLoc> ExprPointsTo;
  // Map from a decl of View type to it pointee. This can be reassigned at
  // various points in the program due to transfer functions.
  llvm::DenseMap<const Decl *, MemoryLoc> DeclPointsTo;

  // Collection of Expr* and Decl* stored on stack.
  llvm::DenseSet<const Expr *> StackExprs;
  llvm::DenseSet<const Decl *> StackDecls;
};
class DanglingReferenceAnalyzer {
public:
  DanglingReferenceAnalyzer(const DeclContext &DC, const CFG &cfg,
                            AnalysisDeclContext &AC,
                            DanglingReferenceReporter *Reporter)
      : DC(DC), cfg(cfg), AC(AC), Reporter(Reporter), PointsTo(DC) {}
  void RunAnalysis() {
    // For simplicity in protoytpe, avoid joins and stick to functions without
    // branches.
    if (!cfg.isLinear())
      return;
    // cfg.dump(AC.getASTContext().getLangOpts(), true);
    for (auto I = cfg.begin(), E = cfg.end(); I != E; ++I) {
      for (CFGBlock::const_iterator BI = (*I)->begin(), BE = (*I)->end();
           BI != BE; ++BI) {
        if (auto cfgstmt = BI->getAs<CFGStmt>()) {
          // llvm::errs() <<
          // "================================================\n";
          // cfgstmt->dump();
          auto *stmt = cfgstmt->getStmt();
          // if (auto *E = dyn_cast<Expr>(stmt))
          //   E->dumpColor();

          PointsTo.Handle(stmt);

          // if (auto *E = dyn_cast<Expr>(stmt)) {
          //   E->dumpColor();
          //   auto Loc = PointsTo.ResolveExpr(E);
          //   llvm::errs() << "Points To : " << Loc.str() << "\n";
          // }

          if (auto *RS = dyn_cast<ReturnStmt>(stmt))
            HandleReturnStmt(RS);
        }
      }
    }
  }

private:
  void HandleReturnStmt(const ReturnStmt *RS) {
    // Diagnose possible return of stack variable.
    if (!RS)
      return;
    const Expr *RetExpr = RS->getRetValue();
    if (!RetExpr || !isPointer(RetExpr))
      return;
    PointsTo.Handle(RetExpr);
    auto RetPointee = PointsTo.ResolveExpr(RetExpr);
    // RetExpr->dumpColor();
    // llvm::errs() << "Returning pointer to " << RetPointee.str() << "\n";
    if (RetPointee.IsOnStack()) {
      llvm::errs() << "=================================================\n";
      llvm::errs() << "Returning pointer to " << RetPointee.str() << "\n\n";
      RetExpr->dumpColor();
      // This points to something on stack!!
      if (auto *D = RetPointee.getDecl())
        Reporter->ReportReturnLocalVar(RetExpr, D);
      if (auto *E = RetPointee.getExpr())
        Reporter->ReportReturnTemporaryExpr(E);
    }
  }

  [[maybe_unused]] const DeclContext &DC;
  const CFG &cfg;
  [[maybe_unused]] AnalysisDeclContext &AC;
  DanglingReferenceReporter *Reporter;
  // ExpressionResolver ExprResolver;
  PointsToTracker PointsTo;
};
} // namespace

void runDanglingReferenceAnalysis(const DeclContext &DC, const CFG &cfg,
                                  AnalysisDeclContext &AC,
                                  DanglingReferenceReporter *Reporter) {
  DanglingReferenceAnalyzer DRA(DC, cfg, AC, Reporter);
  DRA.RunAnalysis();
}

} // namespace clang
