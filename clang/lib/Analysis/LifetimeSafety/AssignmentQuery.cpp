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
#include "llvm/ADT/SmallPtrSet.h"
#include <cstddef>

namespace {

using namespace clang;
using namespace clang::lifetimes;
using namespace clang::lifetimes::internal;

std::optional<const Expr *> GetPureSrcExpr(const Expr *TargetExpr) {
  if (!TargetExpr)
    return std::nullopt;
  const Expr *SExpr = TargetExpr->IgnoreParenCasts();
  if (!SExpr)
    return std::nullopt;

  if (llvm::isa<DeclRefExpr, CXXTemporaryObjectExpr, ConditionalOperator,
                CXXConstructExpr>(SExpr) &&
      !SExpr->getExprLoc().isInvalid())
    return SExpr;

  if (const auto *SCExpr = llvm::dyn_cast<CallExpr>(SExpr);
      SCExpr && !SCExpr->getExprLoc().isInvalid() &&
      !SCExpr->getCallee()->IgnoreParenCasts()->getExprLoc().isInvalid())
    return SCExpr;

  if (const auto *SMExpr = llvm::dyn_cast<MemberExpr>(SExpr))
    return GetPureSrcExpr(SMExpr->getBase());
  if (const auto *SCExpr = llvm::dyn_cast<CXXMemberCallExpr>(SExpr))
    return GetPureSrcExpr(SCExpr->getCallee());
  if (const auto *SUOExpr = llvm::dyn_cast<UnaryOperator>(SExpr))
    return GetPureSrcExpr(SUOExpr->getSubExpr());
  if (const auto *SCBExpr = llvm::dyn_cast<CXXBindTemporaryExpr>(SExpr))
    return GetPureSrcExpr(SCBExpr->getSubExpr());

  return std::nullopt;
}

AliasAssignmentSearchResult getAliasListCore(
    const AssignmentQueryContext &Context, const CFGBlock *Block,
    const LoanID EndLoanID, OriginID *TargetOID,
    const std::optional<OriginDestExpr> LastDestDecl = std::nullopt,
    const std::optional<OriginID> LastOriginID = std::nullopt) {
  std::optional<OriginID> CurrOrigin = std::nullopt;
  std::optional<OriginDestExpr> DestDecl = LastDestDecl;
  std::optional<const Expr *> SrcExpr = std::nullopt;
  llvm::SmallVector<AssignmentPair> AliasStmts;
  const auto Facts = Context.FactMgr.getFacts(Block);
  bool FetchLoan = false;
  auto IssueOriginID = LastOriginID;

  for (const auto &F : llvm::reverse(Facts)) {
    if (const auto *OFF = F->getAs<OriginFlowFact>()) {
      if (IssueOriginID.has_value() &&
          OFF->getDestOriginID() == IssueOriginID.value())
        FetchLoan = true;

      if (OFF->getDestOriginID() == *TargetOID) {
        const auto HeldLoans =
            Context.LoanPropagation.getLoans(OFF->getSrcOriginID(), OFF);

        if (HeldLoans.contains(EndLoanID)) {
          const auto TargetOrigin =
              Context.FactMgr.getOriginMgr().getOrigin(OFF->getDestOriginID());

          if (!DestDecl.has_value()) {
            if (const ValueDecl *DVecl = TargetOrigin.getDecl();
                DVecl && !DVecl->getLocation().isInvalid()) {
              CurrOrigin = *TargetOID;
              DestDecl = DVecl;
            }
          } else {
            auto SExpr = GetPureSrcExpr(TargetOrigin.getExpr());
            if (!SExpr.has_value()) {
              const auto SrcOrigin = Context.FactMgr.getOriginMgr().getOrigin(
                  OFF->getSrcOriginID());
              SExpr = GetPureSrcExpr(SrcOrigin.getExpr());
            }

            if (SExpr.has_value()) {
              AliasStmts.push_back({DestDecl.value(), SExpr.value()});
              SrcExpr = SExpr.value();
              DestDecl = std::nullopt;
              CurrOrigin = std::nullopt;
            }
          }
          *TargetOID = OFF->getSrcOriginID();
        }
      }
    } else if (const auto *IF = F->getAs<IssueFact>()) {
      if (IF->getLoanID() == EndLoanID) {
        IssueOriginID = IF->getOriginID();
      }
    } else if (const auto *UF = F->getAs<UseFact>()) {
      if (CurrOrigin.has_value()) {
        for (const OriginList *Cur = UF->getUsedOrigins(); Cur;
             Cur = Cur->peelOuterOrigin()) {
          if (Cur->getOuterOriginID() == CurrOrigin.value() &&
              UF->isWritten()) {
            const auto UExpr = GetPureSrcExpr(UF->getUseExpr());
            if (UExpr.has_value()) {
              if (const auto *UDExpr =
                      llvm::dyn_cast<DeclRefExpr>(UExpr.value())) {
                DestDecl = UDExpr;
                break;
              }
            }
          }
        }
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
  std::optional<OriginDestExpr> LastDestDecl = std::nullopt;
  llvm::SmallVector<const CFGBlock *> PendingBlocks;
  std::optional<AssignmentPair> StartStmt = std::nullopt;
  std::optional<AssignmentPair> EndStmt = std::nullopt;
  std::optional<OriginID> LastOriginID = std::nullopt;
  llvm::SmallPtrSet<const CFGBlock *, 32> VistedBlocks;
  llvm::DenseMap<AssignmentPair, AssignmentPair> VistedExprs;

  const auto AliasStmtFilter = [&VistedExprs](const AssignmentPair StartStmt,
                                              const AssignmentPair EndStmt) {
    llvm::SmallVector<AssignmentPair> AliasStmts;
    for (auto Stmt = StartStmt; Stmt != EndStmt; Stmt = VistedExprs.at(Stmt))
      AliasStmts.push_back(Stmt);
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
      if (VistedExprs.empty())
        StartStmt = BlockAliasList[0];

      for (size_t i = 0; i < BlockAliasList.size() - 1; ++i)
        VistedExprs.insert({BlockAliasList[i], BlockAliasList[i + 1]});

      if (EndStmt.has_value())
        VistedExprs.insert({EndStmt.value(), BlockAliasList[0]});

      EndStmt = BlockAliasList[BlockAliasList.size() - 1];
    }

    if (Success && StartStmt.has_value() && EndStmt.has_value())
      return AliasStmtFilter(StartStmt.value(), EndStmt.value());

    for (const auto &Block : CurrBlock->preds())
      if (Block && VistedBlocks.insert(Block).second)
        PendingBlocks.push_back(Block);

    if (VistedBlocks.size() >= 32 && StartStmt.has_value() &&
        EndStmt.has_value())
      return AliasStmtFilter(StartStmt.value(), EndStmt.value());
  }

  if (StartStmt.has_value() && EndStmt.has_value())
    return AliasStmtFilter(StartStmt.value(), EndStmt.value());

  return std::nullopt;
}
} // namespace

namespace clang::lifetimes {

ExprPrintingResult FormatIssueExprForSema(const Expr *IssueExpr) {
  if (!IssueExpr)
    return {};
  const auto *PureExpr = IssueExpr->IgnoreParenCasts();
  if (!PureExpr)
    return {};

  if (const auto *IDeclExpr = llvm::dyn_cast<DeclRefExpr>(PureExpr))
    return {FormatValueDeclForSema(IDeclExpr->getDecl()), IssueExpr};
  return {{"the temporary"}, IssueExpr};
}

llvm::SmallVector<ExprPrintingResult>
FormatSrcExprForSema(const Expr *SrcExpr) {
  if (!SrcExpr)
    return {};
  const auto *PureExpr = SrcExpr->IgnoreParenCasts();
  if (!PureExpr)
    return {};

  if (const auto *IOpCallExpr = llvm::dyn_cast<CXXOperatorCallExpr>(PureExpr);
      IOpCallExpr && !IOpCallExpr->getExprLoc().isInvalid())
    return {{{"expression"}, IOpCallExpr->getArg(0)}};
  if (const auto *IDeclExpr = llvm::dyn_cast<DeclRefExpr>(PureExpr);
      IDeclExpr && !IDeclExpr->getExprLoc().isInvalid())
    return {{{}, IDeclExpr}};

  if (const auto *ICXXCallExpr = llvm::dyn_cast<CXXMemberCallExpr>(PureExpr);
      ICXXCallExpr && !ICXXCallExpr->getExprLoc().isInvalid()) {
    llvm::SmallVector<ExprPrintingResult> Result;
    if (!ICXXCallExpr->getCallee()->getExprLoc().isInvalid())
      Result.push_back({{"function call result"}, ICXXCallExpr});
    if (const auto *SubExpr = ICXXCallExpr->getImplicitObjectArgument();
        SubExpr && !llvm::isa<DeclRefExpr>(SubExpr->IgnoreParenCasts())) {
      Result.append(FormatSrcExprForSema(SubExpr));
    }
    return Result;
  }
  if (const auto *ICallExpr = llvm::dyn_cast<CallExpr>(PureExpr);
      ICallExpr && !ICallExpr->getExprLoc().isInvalid()) {
    llvm::SmallVector<ExprPrintingResult> Result;
    if (!ICallExpr->getCallee()->getExprLoc().isInvalid())
      Result.push_back({{"function call result"}, ICallExpr});
    if (const auto *SubExpr = ICallExpr->getCallee();
        SubExpr && !llvm::isa<DeclRefExpr>(SubExpr->IgnoreParenCasts())) {
      Result.append(FormatSrcExprForSema(SubExpr));
    }
    return Result;
  }
  if (const auto *IMemberExpr = llvm::dyn_cast<MemberExpr>(PureExpr);
      IMemberExpr && !IMemberExpr->getExprLoc().isInvalid()) {
    llvm::SmallVector<ExprPrintingResult> Result;
    Result.push_back({{"member access"}, IMemberExpr});
    if (const auto *SubExpr = IMemberExpr->getBase();
        SubExpr && !llvm::isa<DeclRefExpr>(SubExpr->IgnoreParenCasts())) {
      Result.append(FormatSrcExprForSema(SubExpr));
    }
    return Result;
  }

  if (const auto *ICCExpr = llvm::dyn_cast<CXXConstructExpr>(PureExpr)) {
    if (ICCExpr->getNumArgs() > 0) {
      if (const auto *SubExpr = ICCExpr->getArg(0);
          SubExpr && !llvm::isa<DeclRefExpr>(SubExpr->IgnoreParenCasts())) {
        return FormatSrcExprForSema(SubExpr);
      }
    }
  }
  if (const auto *ITempExpr = llvm::dyn_cast<CXXBindTemporaryExpr>(PureExpr)) {
    if (const auto *SubExpr = ITempExpr->getSubExpr();
        SubExpr && !llvm::isa<DeclRefExpr>(SubExpr->IgnoreParenCasts())) {
      return FormatSrcExprForSema(SubExpr);
    }
  }

  return {};
}
} // namespace clang::lifetimes

namespace clang::lifetimes::internal {

std::optional<llvm::SmallVector<AssignmentPair>>
getAliasList(const AssignmentQueryContext &Context, const Fact *CausingFact,
             const LoanID End, const Expr *IssueExpr) {
  llvm::SmallVector<OriginID, 4> TargetOIDList;
  const Expr *WarnningFact = nullptr;

  if (const auto *UF = llvm::dyn_cast<UseFact>(CausingFact)) {
    WarnningFact = UF->getUseExpr();
    for (const OriginList *Cur = UF->getUsedOrigins(); Cur;
         Cur = Cur->peelOuterOrigin())
      TargetOIDList.push_back(Cur->getOuterOriginID());
  } else if (const auto *RetEscapeF =
                 llvm::dyn_cast<ReturnEscapeFact>(CausingFact)) {
    WarnningFact = RetEscapeF->getReturnExpr();
    TargetOIDList.push_back(RetEscapeF->getEscapedOriginID());
  } else {
    llvm_unreachable("Without a corresponding Fact handler, assignment history "
                     "traceback will fail.");
  }

  const CFGBlock *StartBlock =
      Context.ADC.getCFGStmtMap()->getBlock(WarnningFact);
  assert(StartBlock && "Searching CFGBlock failed");
  const CFGBlock *EndBlock = Context.ADC.getCFGStmtMap()->getBlock(IssueExpr);
  assert(EndBlock && "Searching CFGBlock failed");

  for (auto TargetOID : TargetOIDList) {
    if (StartBlock == EndBlock) {
      const AliasAssignmentSearchResult Result =
          getAliasListCore(Context, StartBlock, End, &TargetOID);
      if (!Result.Payload.empty())
        return Result.Payload;
    } else {
      const auto Result =
          getAliasListInMultiBlock(Context, StartBlock, End, &TargetOID);
      if (Result.has_value())
        return Result.value();
    }
  }

  return std::nullopt;
}
} // namespace clang::lifetimes::internal
