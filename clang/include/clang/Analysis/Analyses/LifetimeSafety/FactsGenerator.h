//===- FactsGenerator.h - Lifetime Facts Generation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FactsGenerator, which traverses the AST to generate
// lifetime-relevant facts (such as loan issuance, expiration, origin flow,
// and use) from CFG statements. These facts are used by the dataflow analyses
// to track pointer lifetimes and detect use-after-free errors.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_FACTSGENERATOR_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_FACTSGENERATOR_H

#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/SmallVector.h"

namespace clang::lifetimes::internal {

class FactsGenerator : public ConstStmtVisitor<FactsGenerator> {
  using Base = ConstStmtVisitor<FactsGenerator>;

public:
  FactsGenerator(FactManager &FactMgr, AnalysisDeclContext &AC)
      : FactMgr(FactMgr), AC(AC),
        IsCMode(!AC.getASTContext().getLangOpts().CPlusPlus &&
                !AC.getASTContext().getLangOpts().ObjC) {}

  void run();

  void VisitDeclStmt(const DeclStmt *DS);
  void VisitDeclRefExpr(const DeclRefExpr *DRE);
  void VisitCXXConstructExpr(const CXXConstructExpr *CCE);
  void VisitCXXDefaultInitExpr(const CXXDefaultInitExpr *DIE);
  void VisitCXXMemberCallExpr(const CXXMemberCallExpr *MCE);
  void VisitMemberExpr(const MemberExpr *ME);
  void VisitCallExpr(const CallExpr *CE);
  void VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *N);
  void VisitCastExpr(const CastExpr *CE);
  void VisitUnaryOperator(const UnaryOperator *UO);
  void VisitReturnStmt(const ReturnStmt *RS);
  void VisitBinaryOperator(const BinaryOperator *BO);
  void VisitAbstractConditionalOperator(const AbstractConditionalOperator *CO);
  void VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *OCE);
  void VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *FCE);
  void VisitInitListExpr(const InitListExpr *ILE);
  void VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *BTE);
  void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *MTE);
  void VisitLambdaExpr(const LambdaExpr *LE);
  void VisitArraySubscriptExpr(const ArraySubscriptExpr *ASE);
  void VisitCXXNewExpr(const CXXNewExpr *NE);
  void VisitCXXDeleteExpr(const CXXDeleteExpr *DE);
  void VisitCXXThrowExpr(const CXXThrowExpr *TE);
  void VisitGCCAsmStmt(const GCCAsmStmt *AS);
  void VisitCXXTypeidExpr(const CXXTypeidExpr *TE);
  void VisitStmtExpr(const StmtExpr *SE);

private:
  OriginList *getOriginsList(const ValueDecl &D);
  OriginList *getOriginsList(const Expr &E);

  bool hasOrigins(QualType QT) const;
  bool hasOrigins(const Expr *E) const;

  void flow(OriginList *Dst, OriginList *Src, bool Kill,
            const CFGBlock *Block = nullptr);

  /// Handles assignment for both BinaryOperator and CXXOperatorCallExpr.
  ///
  /// LHSExpr is the destination whose stored loans are replaced by RHSExpr's
  /// loans. TargetExpr is the assignment expression itself; it receives
  /// LHSExpr's origins so chained assignments like `a = b = c` can propagate
  /// through the result of `b = c`.
  void handleAssignment(const Expr *TargetExpr, const Expr *LHSExpr,
                        const Expr *RHSExpr);

  void handlePointerArithmetic(const BinaryOperator *BO);

  bool handlePlacementNew(const CXXNewExpr *NE, OriginList *NewList);

  void handleCXXCtorInitializer(const CXXCtorInitializer *CII);

  void handleLifetimeEnds(const CFGLifetimeEnds &LifetimeEnds);

  void handleFullExprCleanup(const CFGFullExprCleanup &FullExprCleanup);

  void handleExitBlock();

  /// Mark all fields of the implicit object as used for an instance method
  /// call, since the callee may access any part of the object.
  void handleImplicitObjectFieldUses(const Expr *Call, const FunctionDecl *FD);

  void handleGSLPointerConstruction(const CXXConstructExpr *CCE);

  /// Detects arguments passed to rvalue reference parameters and creates
  /// MovedOriginFact for them. The MovedLoansAnalysis then uses these facts
  /// to track in a flow-sensitive manner which loans have been moved at each
  /// program point, allowing warnings to distinguish potentially moved storage
  /// from other use-after-free errors.
  void handleMovedArgsInCall(const FunctionDecl *FD,
                             ArrayRef<const Expr *> Args);

  // Handles [[clang::lifetime_capture_by(X)]] annotations on a function call to
  // create flow facts from captured arguments to the capturer
  void handleLifetimeCaptureBy(const FunctionDecl *FD,
                               ArrayRef<const Expr *> Args);

  /// Checks if a call-like expression creates a borrow by passing a value to a
  /// reference parameter, creating an IssueFact if it does.
  /// \param IsGslConstruction True if this is a GSL construction where all
  ///   argument origins should flow to the returned origin.
  void handleFunctionCall(const Expr *Call, bool IsGslConstruction = false);

  // Detect methods that invalidate iterators/references/pointees.
  // For instance methods, Args[0] is the implicit 'this' pointer.
  void handleInvalidatingCall(const Expr *Call, const FunctionDecl *FD,
                              ArrayRef<const Expr *> Args);

  // Detect explicit destructor calls/`std::destroy_at`
  void handleDestructiveCall(const Expr *Call, const FunctionDecl *FD,
                             ArrayRef<const Expr *> Args);

  template <typename Destination, typename Source>
  void flowOrigin(const Destination &D, const Source &S) {
    flow(getOriginsList(D), getOriginsList(S), /*Kill=*/false);
  }

  template <typename Destination, typename Source>
  void killAndFlowOrigin(const Destination &D, const Source &S) {
    flow(getOriginsList(D), getOriginsList(S), /*Kill=*/true);
  }

  /// Checks if the expression is a `void("__lifetime_test_point_...")` cast.
  /// If so, creates a `TestPointFact` and returns true.
  bool handleTestPoint(const CXXFunctionalCastExpr *FCE);

  /// Whether \p List's outer origin names a declaration's storage outright, so
  /// it can never hold an expired loan (see Origin::NamesDeclStorage).
  bool namesDeclStorage(const OriginList *List) const;

  /// Returns the origins of the value \p E evaluates to, recording the read of
  /// a glvalue. Callers that write to \p E peel the outer origin themselves.
  ///
  /// Example: For `View& v`, returns the origin of what v points to, not v's
  /// storage.
  OriginList *readValue(const Expr *E);

  /// Records an access (read or write) of the storage \p E designates, or that
  /// a prvalue pointer \p E points to.
  void handleAccess(const Expr *E);

  /// Records that \p E's value is handed to opaque code, which may dereference
  /// it to any depth.
  void handleUse(const Expr *E);

  bool escapesViaReturn(OriginID OID) const;

  llvm::SmallVector<Fact *> issuePlaceholderLoans();
  FactManager &FactMgr;
  AnalysisDeclContext &AC;
  llvm::SmallVector<Fact *> CurrentBlockFacts;
  // Collect origins that escape the function in this block (OriginEscapesFact),
  // appended at the end of CurrentBlockFacts to ensure they appear after
  // ExpireFact entries.
  llvm::SmallVector<Fact *> EscapesInCurrentBlock;
  const CFGBlock *CurrentBlock;
  bool IsCMode = false;
};

} // namespace clang::lifetimes::internal

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_FACTSGENERATOR_H
