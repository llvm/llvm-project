//===- AssignmentQuery.cpp - C++ Lifetime Safety Checker --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements trackAssignmentHistory.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/AssignmentQuery.h"
#include <optional>

namespace {
using namespace clang;
using namespace clang::lifetimes;
using namespace clang::lifetimes::internal;

/// Locate the rightmost sub expression of the RHS, given that the LHS is
/// already known. To ensure printability, we invoke `Explorc->isValid()`.
///
/// Typically, we select the rightmost subexpression, as it can be further
/// decomposed and parsed recursively.
///
/// Since we are traversing assignments in reverse order, this function used
/// to determines whether `TargetExpr` meets the requirements of the RHS.
/// A match here triggers a subsequent attempt to match the LHS.
/// Because the function is not re-invoked until the LHS is matched,
/// it generally precludes the possibility of matching multiple
/// subexpressions within the same RHS.
const Expr *getRootSrcExpr(const Expr *TargetExpr) {
  assert(TargetExpr);
  const Expr *SExpr = TargetExpr->IgnoreParenCasts();

  if (isa_and_nonnull<DeclRefExpr, CXXTemporaryObjectExpr, CXXConstructExpr,
                      MemberExpr, CXXMemberCallExpr, UnaryOperator,
                      CXXBindTemporaryExpr>(SExpr) &&
      SExpr->getExprLoc().isValid())
    return SExpr;

  if (const auto *SCExpr = dyn_cast_or_null<CallExpr>(SExpr);
      SCExpr && SCExpr->getExprLoc().isValid() &&
      SCExpr->getCallee()->IgnoreParenCasts()->getExprLoc().isValid())
    return SCExpr;

  return nullptr;
}

/// Obtain the actual LHS Expr from `WriteUF->getUseExpr()` based on the Decl
/// retrieved from `DestOrigin->getDecl()` in the `OriginFlowFact`
DestOriginEntity getDestEntity(const UseFact *UF, const OriginID &OID) {
  for (const OriginList *Cur = UF->getUsedOrigins(); Cur;
       Cur = Cur->peelOuterOrigin()) {
    if (Cur->getOuterOriginID() != OID || !UF->isWritten())
      continue;
    if (const auto *DestExpr =
            dyn_cast_or_null<DeclRefExpr>(getRootSrcExpr(UF->getUseExpr()))) {
      return DestExpr;
    }
  }
  return nullptr;
}

DestOriginEntity getDestEntity(const FactManager &FactMgr,
                               const OriginFlowFact *OFF) {
  const Origin &DestOrigin =
      FactMgr.getOriginMgr().getOrigin(OFF->getDestOriginID());
  const Origin &SrcOrigin =
      FactMgr.getOriginMgr().getOrigin(OFF->getSrcOriginID());

  if (const ValueDecl *DestDecl = DestOrigin.getDecl();
      DestDecl && DestDecl->getLocation().isValid()) {
    return DestDecl;
  }

  // This logic specifically handles the isa<FieldDecl>(DestOrigin->getDecl())
  // case. In `OriginFlowFact`, we store the Decl of the corresponding variable
  // as the Origin rather than the LHS Origin itself.
  //
  // For a general `ValueDecl`, we typically find the corresponding `UseFact`
  // following the `OriginFlowFact`. However, for a `FieldDecl`, the subsequent
  // `OriginFlowFact` is associated with a `MemberExpr`. In this scenario,
  // `DestOrigin` represents the `MemberExpr`, while SrcOrigin represents the
  // Origin of the `CXXThisExpr` (CXXMethodDecl).
  const Expr *DestExpr = DestOrigin.getExpr();
  const ValueDecl *SrcDecl = SrcOrigin.getDecl();
  if (isa_and_nonnull<MemberExpr>(DestExpr) &&
      isa_and_nonnull<CXXMethodDecl>(SrcDecl))
    return dyn_cast<MemberExpr>(DestExpr);

  return nullptr;
}

SrcOriginEntity getSrcEntity(const FactManager &FactMgr,
                             const OriginFlowFact *OFF) {
  const Origin &DestOrigin =
      FactMgr.getOriginMgr().getOrigin(OFF->getDestOriginID());

  const Expr *SExpr = getRootSrcExpr(DestOrigin.getExpr());
  if (!SExpr) {
    const Origin &SrcOrigin =
        FactMgr.getOriginMgr().getOrigin(OFF->getSrcOriginID());
    SExpr = getRootSrcExpr(SrcOrigin.getExpr());
  }

  return SExpr;
}

bool trackAssignmentHistoryCore(
    const FactManager &FactMgr,
    const LoanPropagationAnalysis &LoanPropagation,
    llvm::SmallVectorImpl<AssignmentPair> &AssignmentList,
    const CFGBlock *Block, OriginID *TargetOID, const LoanID EndLoanID) {
  DestOriginEntity CurrDestEntity = nullptr;
  bool NeedSearchOriginDestWithoutLoan = false;
  std::optional<OriginID> CurrOriginID = std::nullopt;
  llvm::ArrayRef<const Fact *> Facts = FactMgr.getFacts(Block);

  const auto TryInsertAssignmentList = [&](const OriginFlowFact *OFF) {
    if (NeedSearchOriginDestWithoutLoan) {
      if (const MemberExpr *DestMemberExpr =
              dyn_cast_or_null<const MemberExpr *>(
                  getDestEntity(FactMgr, OFF))) {
        CurrDestEntity = DestMemberExpr;
        NeedSearchOriginDestWithoutLoan = false;
      }
    }

    if (OFF->getDestOriginID() == *TargetOID &&
        LoanPropagation.getLoans(OFF->getSrcOriginID(), OFF)
            .contains(EndLoanID)) {
      if (!CurrDestEntity) {
        DestOriginEntity DestEntity = getDestEntity(FactMgr, OFF);
        auto *DestValueDecl = dyn_cast_or_null<const ValueDecl *>(DestEntity);
        if (DestValueDecl)
          CurrOriginID = *TargetOID;

        if (llvm::isa_and_nonnull<FieldDecl>(DestValueDecl))
          NeedSearchOriginDestWithoutLoan = true;
        else
          CurrDestEntity = DestEntity;
      } else {
        SrcOriginEntity CurrSrcEntity = getSrcEntity(FactMgr, OFF);
        if (CurrSrcEntity) {
          AssignmentList.push_back({CurrDestEntity, CurrSrcEntity});
          CurrDestEntity = nullptr;
          CurrOriginID = std::nullopt;
        }
      }
      *TargetOID = OFF->getSrcOriginID();
    }
  };

  for (const Fact *F : llvm::reverse(Facts)) {
    if (const auto *OFF = F->getAs<OriginFlowFact>()) {
      TryInsertAssignmentList(OFF);
    } else if (const auto *IF = F->getAs<IssueFact>()) {
      if (IF->getLoanID() == EndLoanID)
        return true;
    } else if (const auto *UF = F->getAs<UseFact>()) {
      if (CurrOriginID) {
        DestOriginEntity DestEntity = getDestEntity(UF, CurrOriginID.value());
        if (DestEntity)
          CurrDestEntity = DestEntity;
      }
    }
  }

  return false;
}
} // namespace

namespace clang::lifetimes::internal {

void trackAssignmentHistory(
    const FactManager &FactMgr,
    const LoanPropagationAnalysis &LoanPropagation,
    llvm::SmallVectorImpl<AssignmentPair> &AssignmentList,
    const CFGBlock *StartBlock, OriginID StartOID, const LoanID EndLoanID) {
  if (!trackAssignmentHistoryCore(FactMgr, LoanPropagation, AssignmentList, StartBlock,
                                  &StartOID, EndLoanID))
    llvm::errs() << "Assignment History Tracking may have failed\n";
  std::reverse(AssignmentList.begin(), AssignmentList.end());
}
} // namespace clang::lifetimes::internal
