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

  return Result;
}
bool isOwner(const Expr *E) {
  return isRecordWithAttr<OwnerAttr>(E->getType());
}
bool isOwner(const Decl *D) {
  return isa<ValueDecl>(D) &&
         isRecordWithAttr<OwnerAttr>(dyn_cast<ValueDecl>(D)->getType());
}
bool isPointer(const Expr *E) {
  return isRecordWithAttr<PointerAttr>(E->getType());
}
bool isPointer(const Decl *D) {
  return isa<ValueDecl>(D) &&
         isRecordWithAttr<PointerAttr>(dyn_cast<ValueDecl>(D)->getType());
}

struct MemoryLoc {
  enum MemoryType {
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

class PointsToTracker : public ConstStmtVisitor<PointsToTracker> {
public:
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
      if (DeclPointsTo.find(D) == DeclPointsTo.end())
        UpdatePointer(VD, ResolveExpr(VD->getInit()));
    }
    if (isOwner(VD)) {
      if (StackDecls.find(D) == StackDecls.end() && VD->hasLocalStorage())
        AddToStack(VD);
    }
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
    }
    if (isOwner(MTE)) {
      // We have a temporary owner on stack.
      AddToStack(MTE);
    }
  }

  void VisitImplicitCastExpr(const ImplicitCastExpr *E) {
    auto *SE = E->IgnoreImpCasts();
    Handle(SE);
    SetExprPointer(E, SE);
  }

  void VisitExprWithCleanups(const ExprWithCleanups *E) {
    // Handle(E->getSubExpr());
    SetExprPointer(E, E->getSubExpr());
  }

  void VisitCXXMemberCallExpr(const CXXMemberCallExpr *MCE) {
    // Conversion from Owner to a Pointer.
    const Expr *ConversionFrom = MCE->IgnoreConversionOperatorSingleStep();
    if (ConversionFrom != MCE) {
      if (isOwner(ConversionFrom) && isPointer(MCE)) {
        SetExprPointer(MCE, ConversionFrom);
      }
    }
  }

  void VisitCXXConstructExpr(const CXXConstructExpr *CCE) {
    if (!isPointer(CCE))
      return;
    if (CCE->getNumArgs() == 1)
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

    if (const Decl *PointerDecl = DeclReferencedBy(A))
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
    assert(ExprPointsTo.insert({E, ML}).second);
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
    Handle(E);
    auto Res = ExprPointsTo.find(E);
    if (Res != ExprPointsTo.end()) {
      return Res->getSecond();
    }
    return ExprPointsTo[E] = MemoryLoc::Unknown();
  }

  // Returns the memory location pointed to by D. If D is a pointer-type,
  // returns the memory pointed to by the pointer.
  MemoryLoc ResolveDecl(const Decl *D) {
    MaybeInitaliseDecl(D);
    if (isPointer(D))
      return ResolvePointer(D);
    if (isOwner(D))
      return ResolveOwner(D);
    return MemoryLoc::Unknown();
  }

  MemoryLoc ResolvePointer(const Decl *D) {
    auto *VD = dyn_cast<VarDecl>(D);
    assert(VD);
    if (!VD->hasLocalStorage())
      return MemoryLoc::Unknown();
    assert(isPointer(D));
    auto Res = DeclPointsTo.find(D);
    assert(Res != DeclPointsTo.end());
    return Res->getSecond();
  }

  MemoryLoc ResolveOwner(const Decl *D) {
    assert(isOwner(D));
    if (IsOnStack(D))
      return MemoryLoc::VarOnStack(D);
    return MemoryLoc::Unknown();
  }

  void AddToStack(const Decl *D) {
    assert(isOwner(D));
    StackDecls.insert(D);
  }
  void AddToStack(const Expr *E) {
    assert(isa<MaterializeTemporaryExpr>(E));
    assert(isOwner(E));
    StackExprs.insert(E);
    // Add a self edge.
    assert(ExprPointsTo.insert({E, MemoryLoc::Temporary(E)}).second);
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
      : DC(DC), cfg(cfg), AC(AC), Reporter(Reporter) {}
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
          auto *stmt = cfgstmt->getStmt();
          // llvm::errs() <<
          // "================================================\n";
          // cfgstmt->dump();
          // if (auto *E = dyn_cast<Expr>(stmt))
          //   E->dumpColor();
          PointsTo.Handle(stmt);
          if (auto *E = dyn_cast<Expr>(stmt)) {
            // llvm::errs() << "Points To : " << PointsTo.ResolveExpr(E).str()
            //              << "\n";
          }
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
