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
#include "clang/AST/ParentMap.h"
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

/// Specifically handles assignments involving a FieldDecl.
///
/// Since we currently only store the FieldDecl without its corresponding
/// LHS expression, this function attempts to recover or resolve the LHS
/// context by analyzing the RHS.
const MemberExpr *getFieldFromAssignmentExpr(const Expr *RHS,
                                             const ParentMap &CurrParentMap) {

  const Stmt *CurrStmt = CurrParentMap.getParent(RHS);
  if (!CurrStmt)
    return nullptr;
  if (const auto *BinaryOp = llvm::dyn_cast<BinaryOperator>(CurrStmt))
    return llvm::dyn_cast<MemberExpr>(BinaryOp->getLHS());
  if (const auto *CXXOp = llvm::dyn_cast<CXXOperatorCallExpr>(CurrStmt);
      CXXOp && CXXOp->getOperator() == OO_Equal && CXXOp->getNumArgs() == 2)
    return llvm::dyn_cast<MemberExpr>(CXXOp->getArg(0));
  return nullptr;
}

AliasAssignmentSearchResult getAliasListCore(
    const AssignmentQueryContext &Context, const CFGBlock *Block,
    const LoanID EndLoanID, OriginID *TargetOID,
    const std::optional<OriginDestExpr> LastDestExpr = std::nullopt,
    const std::optional<OriginID> LastOriginID = std::nullopt) {
  llvm::SmallVector<AssignmentPair> AliasStmts;
  llvm::ArrayRef<const Fact *> Facts = Context.FactMgr.getFacts(Block);
  std::optional<OriginID> IssueOriginID = LastOriginID;
  std::optional<OriginDestExpr> DestExpr = LastDestExpr;
  std::optional<OriginID> CurrOrigin = std::nullopt;
  std::optional<const Expr *> SrcExpr = std::nullopt;

  for (const Fact *F : llvm::reverse(Facts)) {
    if (const auto *OFF = F->getAs<OriginFlowFact>()) {
      if (IssueOriginID && OFF->getDestOriginID() == IssueOriginID.value())
        return {AliasStmts, true, DestExpr, IssueOriginID};
      if (OFF->getDestOriginID() == *TargetOID) {
        const LoanSet HeldLoans =
            Context.LoanPropagation.getLoans(OFF->getSrcOriginID(), OFF);

        if (HeldLoans.contains(EndLoanID)) {
          const Origin TargetOrigin =
              Context.FactMgr.getOriginMgr().getOrigin(OFF->getDestOriginID());

          if (!DestExpr.has_value()) {
            if (const ValueDecl *DVecl = TargetOrigin.getDecl();
                DVecl && !DVecl->getLocation().isInvalid()) {
              if (llvm::isa<FieldDecl>(DVecl)) {
                const auto *CurrExpr = Context.FactMgr.getOriginMgr()
                                           .getOrigin(OFF->getSrcOriginID())
                                           .getExpr();
                if (CurrExpr)
                  DestExpr = getFieldFromAssignmentExpr(
                      CurrExpr, Context.ADC.getParentMap());
              } else {
                CurrOrigin = *TargetOID;
                DestExpr = DVecl;
              }
            }
          } else {
            auto SExpr = GetPureSrcExpr(TargetOrigin.getExpr());
            if (!SExpr) {
              const auto SrcOrigin = Context.FactMgr.getOriginMgr().getOrigin(
                  OFF->getSrcOriginID());
              SExpr = GetPureSrcExpr(SrcOrigin.getExpr());
            }

            if (SExpr) {
              AliasStmts.push_back({DestExpr.value(), SExpr.value()});
              SrcExpr = SExpr.value();
              DestExpr = std::nullopt;
              CurrOrigin = std::nullopt;
            }
          }
          *TargetOID = OFF->getSrcOriginID();
        }
      }
    } else if (const auto *IF = F->getAs<IssueFact>()) {
      if (IF->getLoanID() == EndLoanID)
        IssueOriginID = IF->getOriginID();
    } else if (const auto *UF = F->getAs<UseFact>()) {
      if (CurrOrigin.has_value()) {
        for (const OriginList *Cur = UF->getUsedOrigins(); Cur;
             Cur = Cur->peelOuterOrigin()) {
          if (Cur->getOuterOriginID() != CurrOrigin.value() || !UF->isWritten())
            continue;
          std::optional<const Expr *> UExpr = GetPureSrcExpr(UF->getUseExpr());
          if (UExpr.has_value()) {
            if (const auto *UDExpr =
                    llvm::dyn_cast<DeclRefExpr>(UExpr.value())) {
              DestExpr = UDExpr;
              break;
            }
          }
        }
      }
    }
  }
  return {AliasStmts, false, DestExpr, IssueOriginID};
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

void FormatRHSValueDeclForSema(const ValueDecl *TargetValue,
                               llvm::SmallVectorImpl<char> &IssueMsg) {
  if (TargetValue) {
    const StringRef TargetName = TargetValue->getName();
    IssueMsg.push_back('\'');
    IssueMsg.append(TargetName.begin(), TargetName.end());
    IssueMsg.push_back('\'');
  }
}
} // namespace

namespace clang::lifetimes {

void FormatLoanEntityForSema(LoanEntity IssueEntity,
                             llvm::SmallVectorImpl<char> &IssueMsg) {
  llvm::StringRef Temp = "the temporary";

  if (const auto *IssueExpr = llvm::dyn_cast<const Expr *>(IssueEntity)) {
    if (const auto *IDeclExpr =
            llvm::dyn_cast<DeclRefExpr>(IssueExpr->IgnoreParenCasts()))
      FormatRHSValueDeclForSema(IDeclExpr->getDecl(), IssueMsg);
  }

  if (const auto *IssueParmDecl =
          llvm::dyn_cast<const ParmVarDecl *>(IssueEntity))
    FormatRHSValueDeclForSema(IssueParmDecl, IssueMsg);
  else if (const auto *IssueCXXMD =
               llvm::dyn_cast<const CXXMethodDecl *>(IssueEntity))
    FormatRHSValueDeclForSema(IssueCXXMD, IssueMsg);
  else
    IssueMsg.append(Temp.begin(), Temp.end());
}

void FormatSrcExprForSema(
    const Expr *SrcExpr,
    llvm::SmallVectorImpl<ExprPrintingResult> &SrcMsgList) {
  if (!SrcExpr)
    return;
  const auto *PureExpr = SrcExpr->IgnoreParenCasts();
  if (!PureExpr)
    return;

  const auto AppendSubExprIfNeeded = [&SrcMsgList](const Expr *SubExpr) {
    if (SubExpr && !llvm::isa<DeclRefExpr>(SubExpr->IgnoreParenCasts())) {
      FormatSrcExprForSema(SubExpr, SrcMsgList);
    }
  };

  if (const auto *IOpCallExpr = llvm::dyn_cast<CXXOperatorCallExpr>(PureExpr);
      IOpCallExpr && !IOpCallExpr->getExprLoc().isInvalid()) {
    SrcMsgList.push_back({"expression", IOpCallExpr->getArg(0)});
  } else if (const auto *IDeclExpr = llvm::dyn_cast<DeclRefExpr>(PureExpr);
             IDeclExpr && !IDeclExpr->getExprLoc().isInvalid()) {
    SrcMsgList.push_back({"", IDeclExpr});
  } else if (const auto *ICXXCallExpr =
                 llvm::dyn_cast<CXXMemberCallExpr>(PureExpr);
             ICXXCallExpr && !ICXXCallExpr->getExprLoc().isInvalid()) {
    if (!ICXXCallExpr->getCallee()->getExprLoc().isInvalid())
      SrcMsgList.push_back({"function call result", ICXXCallExpr});
    AppendSubExprIfNeeded(ICXXCallExpr->getImplicitObjectArgument());
  } else if (const auto *ICallExpr = llvm::dyn_cast<CallExpr>(PureExpr);
             ICallExpr && !ICallExpr->getExprLoc().isInvalid()) {
    if (!ICallExpr->getCallee()->getExprLoc().isInvalid())
      SrcMsgList.push_back({"function call result", ICallExpr});
    AppendSubExprIfNeeded(ICallExpr->getCallee());
  } else if (const auto *IMemberExpr = llvm::dyn_cast<MemberExpr>(PureExpr);
             IMemberExpr && !IMemberExpr->getExprLoc().isInvalid()) {
    SrcMsgList.push_back({"member access", IMemberExpr});
    AppendSubExprIfNeeded(IMemberExpr->getBase());
  } else if (const auto *ICCExpr = llvm::dyn_cast<CXXConstructExpr>(PureExpr);
             ICCExpr && ICCExpr->getNumArgs() > 0) {
    AppendSubExprIfNeeded(ICCExpr->getArg(0));
  } else if (const auto *ITempExpr =
                 llvm::dyn_cast<CXXBindTemporaryExpr>(PureExpr)) {
    AppendSubExprIfNeeded(ITempExpr->getSubExpr());
  }
}
} // namespace clang::lifetimes

namespace clang::lifetimes::internal {

void getAliasList(const AssignmentQueryContext &Context,
                  llvm::SmallVectorImpl<AssignmentPair> &AssignmentList,
                  const Fact *CausingFact, const LoanID End,
                  const CFGBlock *StartBlock, const Expr *IssueExpr) {
  llvm::SmallVector<OriginID, 4> TargetOIDList;

  if (const auto *UF = llvm::dyn_cast<UseFact>(CausingFact))
    for (const OriginList *Cur = UF->getUsedOrigins(); Cur;
         Cur = Cur->peelOuterOrigin())
      TargetOIDList.push_back(Cur->getOuterOriginID());
  else if (const auto *RetEscapeF =
               llvm::dyn_cast<ReturnEscapeFact>(CausingFact))
    TargetOIDList.push_back(RetEscapeF->getEscapedOriginID());
  else if (const auto *FieldEscapeF =
               llvm::dyn_cast<FieldEscapeFact>(CausingFact))
    TargetOIDList.push_back(FieldEscapeF->getEscapedOriginID());
  else if (const auto *GlobalEscapeF =
               llvm::dyn_cast<GlobalEscapeFact>(CausingFact))
    TargetOIDList.push_back(GlobalEscapeF->getEscapedOriginID());
  else
    llvm_unreachable("Without a corresponding Fact handler, assignment history "
                     "traceback will fail.");

  const CFGBlock *EndBlock =
      IssueExpr ? Context.ADC.getCFGStmtMap()->getBlock(IssueExpr) : nullptr;

  for (OriginID TargetOID : TargetOIDList) {
    if (StartBlock == EndBlock) {
      AliasAssignmentSearchResult Result =
          getAliasListCore(Context, StartBlock, End, &TargetOID);
      if (!Result.Payload.empty())
        AssignmentList = Result.Payload;
    } else {
      std::optional<llvm::SmallVector<AssignmentPair>> Result =
          getAliasListInMultiBlock(Context, StartBlock, End, &TargetOID);
      if (Result)
        AssignmentList = Result.value();
    }
  }
}
} // namespace clang::lifetimes::internal
