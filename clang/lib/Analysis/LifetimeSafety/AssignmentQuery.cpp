//===- AssignmentQuery.cpp - C++ Lifetime Safety Checker --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LifetimeChecker, which detects use-after-free
// errors by checking if live origins hold loans that have expired.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/AssignmentQuery.h"
#include "clang/AST/Decl.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include <cstddef>

namespace {

using namespace clang;
using namespace clang::lifetimes;
using namespace clang::lifetimes::internal;

std::optional<OriginSrcExpr> GetPureSrcExpr(const Expr *TargetExpr) {
  if (!TargetExpr)
    return std::nullopt;
  const Expr *SExpr = TargetExpr->IgnoreParenCasts();
  if (!SExpr)
    return std::nullopt;

  if (const auto *SDRExpr = llvm::dyn_cast<DeclRefExpr>(SExpr)) {
    return SDRExpr;
  }
  if (const auto *STMExpr = llvm::dyn_cast<CXXTemporaryObjectExpr>(SExpr)) {
    return STMExpr;
  }
  if (const auto *SCExpr = llvm::dyn_cast<CallExpr>(SExpr)) {
    return SCExpr;
  }

  if (const auto *SCCExpr = llvm::dyn_cast<CXXConstructExpr>(SExpr)) {
    if (SCCExpr->getNumArgs() > 0)
      return GetPureSrcExpr(SCCExpr->getArg(0));
  }
  if (const auto *SUOExpr = llvm::dyn_cast<UnaryOperator>(SExpr)) {
    return GetPureSrcExpr(SUOExpr->getSubExpr());
  }
  if (const auto *SCBExpr = llvm::dyn_cast<CXXBindTemporaryExpr>(SExpr)) {
    return GetPureSrcExpr(SCBExpr->getSubExpr());
  }

  return std::nullopt;
}

AliasAssignmentSearchResult
getAliasListCore(const AssignmentQueryContext &Context, const CFGBlock *Block,
                 const LoanID EndLoanID, OriginID *TargetOID,
                 const ValueDecl *LastDestDecl = nullptr,
                 const std::optional<OriginID> LastOriginID = std::nullopt) {
  llvm::SmallVector<AssignmentPair> AliasStmts;
  const ValueDecl *DestDecl = LastDestDecl;
  const auto Facts = Context.FactMgr.getFacts(Block);
  bool FetchLoan = false;
  auto IssueOriginID = LastOriginID;

  for (auto F = Facts.rbegin(); F != Facts.rend(); ++F) {
    if (const auto *OFF = (*F)->getAs<OriginFlowFact>()) {
      if (IssueOriginID.has_value() &&
          OFF->getDestOriginID() == IssueOriginID.value()) {
        FetchLoan = true;
      }
      if (OFF->getDestOriginID() == *TargetOID) {
        const auto HeldLoans =
            Context.LoanPropagation.getLoans(OFF->getSrcOriginID(), OFF);

        if (HeldLoans.contains(EndLoanID)) {
          const auto TargetOrigin =
              Context.FactMgr.getOriginMgr().getOrigin(OFF->getDestOriginID());

          if (DestDecl == nullptr) {
            if (const ValueDecl *DDecl = TargetOrigin.getDecl()) {
              DestDecl = DDecl;
            }
          } else {
            auto SExpr = GetPureSrcExpr(TargetOrigin.getExpr());
            if (!SExpr.has_value()) {
              const auto SrcOrigin = Context.FactMgr.getOriginMgr().getOrigin(
                  OFF->getSrcOriginID());
              SExpr = GetPureSrcExpr(SrcOrigin.getExpr());
            }

            if (SExpr.has_value()) {
              AliasStmts.push_back({SExpr.value(), DestDecl});
              DestDecl = nullptr;
            }
          }
          *TargetOID = OFF->getSrcOriginID();
        }
      }
    } else if (const auto *IF = (*F)->getAs<IssueFact>()) {
      if (IF->getLoanID() == EndLoanID) {
        IssueOriginID = IF->getOriginID();
      }
    }

    if (FetchLoan) {
      return {AliasStmts, true, DestDecl, IssueOriginID};
    }
  }
  return {AliasStmts, false, DestDecl, IssueOriginID};
}

std::optional<llvm::SmallVector<AssignmentPair>>
getAliasListInMultiBlock(const AssignmentQueryContext &Context,
                         const CFGBlock *StartBlock, const LoanID EndLoanID,
                         OriginID *StartOID) {
  const ValueDecl *LastDestDecl = nullptr;
  llvm::SmallVector<const CFGBlock *> PendingBlocks;
  std::optional<AssignmentPair> StartStmt = std::nullopt;
  std::optional<AssignmentPair> EndStmt = std::nullopt;
  std::optional<OriginID> LastOriginID = std::nullopt;
  llvm::SmallPtrSet<const CFGBlock *, 32> VistedBlocks;
  llvm::DenseMap<AssignmentPair, AssignmentPair> VistedExprs;

  const auto AliasStmtFilter = [&VistedExprs](const AssignmentPair StartStmt,
                                              const AssignmentPair EndStmt) {
    llvm::SmallVector<AssignmentPair> AliasStmts;
    for (auto Stmt = StartStmt; Stmt != EndStmt; Stmt = VistedExprs.at(Stmt)) {
      AliasStmts.push_back(Stmt);
    }
    AliasStmts.push_back(EndStmt);
    return AliasStmts;
  };

  PendingBlocks.push_back(StartBlock);

  for (size_t i = 0; i < PendingBlocks.size(); ++i) {
    const CFGBlock *CurrBlock = PendingBlocks[i];

    const auto [BlockAliasList, Success, CurrLastDestDecl, CurrLastOriginID] =
        getAliasListCore(Context, CurrBlock, EndLoanID, StartOID, LastDestDecl,
                         LastOriginID);
    if (CurrLastDestDecl)
      LastDestDecl = CurrLastDestDecl;
    if (CurrLastOriginID.has_value())
      LastOriginID = CurrLastOriginID;

    if (!BlockAliasList.empty()) {
      if (VistedExprs.empty()) {
        StartStmt = BlockAliasList[0];
      }

      for (size_t i = 0; i < BlockAliasList.size() - 1; ++i) {
        VistedExprs.insert({BlockAliasList[i], BlockAliasList[i + 1]});
      }

      if (EndStmt.has_value())
        VistedExprs.insert({EndStmt.value(), BlockAliasList[0]});

      EndStmt = BlockAliasList[BlockAliasList.size() - 1];
    }

    if (Success && StartStmt.has_value() && EndStmt.has_value()) {
      return AliasStmtFilter(StartStmt.value(), EndStmt.value());
    }

    for (const auto Block : CurrBlock->preds()) {
      if (Block && VistedBlocks.insert(Block).second)
        PendingBlocks.push_back(Block);
    }

    if (VistedBlocks.size() >= 32 && StartStmt.has_value() &&
        EndStmt.has_value()) {
      return AliasStmtFilter(StartStmt.value(), EndStmt.value());
    }
  }

  if (StartStmt.has_value() && EndStmt.has_value()) {
    return AliasStmtFilter(StartStmt.value(), EndStmt.value());
  }

  return std::nullopt;
}
} // namespace

namespace clang::lifetimes::internal {

std::optional<llvm::SmallVector<AssignmentPair>>
getAliasList(const AssignmentQueryContext &Context, const UseFact *UF,
             const LoanID End, const bool InOneBlock) {
  const CFGBlock *IssueBlock =
      Context.ADC.getCFGStmtMap()->getBlock(UF->getUseExpr());
  assert(IssueBlock && "Searching CFGBlock failed");

  for (const OriginList *Cur = UF->getUsedOrigins(); Cur;
       Cur = Cur->peelOuterOrigin()) {
    auto TargetOID = Cur->getOuterOriginID();
    if (InOneBlock) {
      AliasAssignmentSearchResult Result =
          getAliasListCore(Context, IssueBlock, End, &TargetOID);
      if (!Result.Payload.empty())
        return Result.Payload;
    } else {
      auto Result =
          getAliasListInMultiBlock(Context, IssueBlock, End, &TargetOID);
      if (Result.has_value())
        return Result.value();
    }
  }
  return std::nullopt;
}
} // namespace clang::lifetimes::internal
