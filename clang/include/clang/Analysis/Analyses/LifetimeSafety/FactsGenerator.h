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
      : FactMgr(FactMgr), AC(AC) {}

  void run();

  void VisitDeclStmt(const DeclStmt *DS);
  void VisitDeclRefExpr(const DeclRefExpr *DRE);
  void VisitCXXConstructExpr(const CXXConstructExpr *CCE);
  void VisitCXXMemberCallExpr(const CXXMemberCallExpr *MCE);
  void VisitCallExpr(const CallExpr *CE);
  void VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *N);
  void VisitImplicitCastExpr(const ImplicitCastExpr *ICE);
  void VisitUnaryOperator(const UnaryOperator *UO);
  void VisitReturnStmt(const ReturnStmt *RS);
  void VisitBinaryOperator(const BinaryOperator *BO);
  void VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *OCE);
  void VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *FCE);
  void VisitInitListExpr(const InitListExpr *ILE);
  void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *MTE);

private:
  void handleDestructor(const CFGAutomaticObjDtor &DtorOpt);

  void handleGSLPointerConstruction(const CXXConstructExpr *CCE);

  /// Checks if a call-like expression creates a borrow by passing a value to a
  /// reference parameter, creating an IssueFact if it does.
  /// \param IsGslConstruction True if this is a GSL construction where all
  ///   argument origins should flow to the returned origin.
  void handleFunctionCall(const Expr *Call, const FunctionDecl *FD,
                          ArrayRef<const Expr *> Args,
                          bool IsGslConstruction = false);

  template <typename Destination, typename Source>
  void flowOrigin(const Destination &D, const Source &S) {
    OriginID DestOID = FactMgr.getOriginMgr().getOrCreate(D);
    OriginID SrcOID = FactMgr.getOriginMgr().get(S);
    CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
        DestOID, SrcOID, /*KillDest=*/false));
  }

  template <typename Destination, typename Source>
  void killAndFlowOrigin(const Destination &D, const Source &S) {
    OriginID DestOID = FactMgr.getOriginMgr().getOrCreate(D);
    OriginID SrcOID = FactMgr.getOriginMgr().get(S);
    CurrentBlockFacts.push_back(
        FactMgr.createFact<OriginFlowFact>(DestOID, SrcOID, /*KillDest=*/true));
  }

  /// Checks if the expression is a `void("__lifetime_test_point_...")` cast.
  /// If so, creates a `TestPointFact` and returns true.
  bool handleTestPoint(const CXXFunctionalCastExpr *FCE);

  void handleAssignment(const Expr *LHSExpr, const Expr *RHSExpr);

  // A DeclRefExpr will be treated as a use of the referenced decl. It will be
  // checked for use-after-free unless it is later marked as being written to
  // (e.g. on the left-hand side of an assignment).
  void handleUse(const DeclRefExpr *DRE);

  void markUseAsWrite(const DeclRefExpr *DRE);

  FactManager &FactMgr;
  AnalysisDeclContext &AC;
  llvm::SmallVector<Fact *> CurrentBlockFacts;
  // To distinguish between reads and writes for use-after-free checks, this map
  // stores the `UseFact` for each `DeclRefExpr`. We initially identify all
  // `DeclRefExpr`s as "read" uses. When an assignment is processed, the use
  // corresponding to the left-hand side is updated to be a "write", thereby
  // exempting it from the check.
  llvm::DenseMap<const DeclRefExpr *, UseFact *> UseFacts;
};

} // namespace clang::lifetimes::internal

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_FACTSGENERATOR_H
