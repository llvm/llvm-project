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
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/SmallSet.h"
#include <cstddef>
#include <queue>

namespace {

using namespace clang;
using namespace clang::lifetimes;
using namespace clang::lifetimes::internal;

struct AssignmentSearchToken {
  const CFGBlock *Block;
  const OriginID OID;

  bool operator==(const AssignmentSearchToken& Other) const {
    return Block == Other.Block && OID == Other.OID;
  }
  bool operator<(const AssignmentSearchToken& Other) const {
    if (Block == Other.Block)
      return OID < Other.OID;
    return Block < Other.Block;
  }
};

struct AssignmentSearchContext {
  const AssignmentSearchToken CurrToken;
  const OriginDestExpr LastDestDeclOrExpr;
  const std::optional<AssignmentPair> LastEndAssignment;
  const std::optional<OriginID> LastOriginID;
};

/// Locate the rightmost sub expression of the RHS, given that the LHS is
/// already known. To ensure printability, we invoke `Explorc->isValid()`.
///
/// `PureSrcExpr` refers to the root expression representing the RHS.
/// Generally, we aim for the rightmost sub expression because it will be
/// recursively parsed in `formatSrcExprForSema`.
///
/// Most expressions accepted by this function have corresponding handling logic
/// in `formatSrcExprForSema`.
///
/// Note that the returned `DeclRefExpr` acts purely as a placeholder.
/// For simple assignments like `a = b`, we do not need to provide source
/// location highlights for the RHS, so `formatSrcExprForSema` will not perform
/// any meaningful parsing on this `DeclRefExpr`.
const Expr *getPureSrcExpr(const Expr *TargetExpr) {
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
const DeclRefExpr *getLHSExpr(const UseFact *UF, const OriginID OID) {
  for (const OriginList *Cur = UF->getUsedOrigins(); Cur;
       Cur = Cur->peelOuterOrigin()) {
    if (Cur->getOuterOriginID() != OID || !UF->isWritten())
      continue;
    if (const auto *UDestExpr =
            dyn_cast_or_null<DeclRefExpr>(getPureSrcExpr(UF->getUseExpr()))) {
      return UDestExpr;
    }
  }
  return nullptr;
}

OriginDestExpr getLHSDeclOrExpr(const AssignmentQueryContext &Context,
                                const OriginFlowFact *OFF) {
  const Origin &DestOrigin =
      Context.FactMgr.getOriginMgr().getOrigin(OFF->getDestOriginID());
  const Origin &SrcOrigin =
      Context.FactMgr.getOriginMgr().getOrigin(OFF->getSrcOriginID());

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

const Expr *getRHSExpr(const AssignmentQueryContext &Context,
                       const OriginFlowFact *OFF) {
  const Origin &DestOrigin =
      Context.FactMgr.getOriginMgr().getOrigin(OFF->getDestOriginID());

  const Expr *SExpr = getPureSrcExpr(DestOrigin.getExpr());
  if (!SExpr) {
    const Origin &SrcOrigin =
        Context.FactMgr.getOriginMgr().getOrigin(OFF->getSrcOriginID());
    SExpr = getPureSrcExpr(SrcOrigin.getExpr());
  }

  return SExpr;
}

AliasAssignmentSearchResult
getAliasListCore(const AssignmentQueryContext &Context,
                 llvm::SmallVectorImpl<AssignmentPair> &AssignmentList,
                 const CFGBlock *Block, const LoanID EndLoanID,
                 OriginID *TargetOID,
                 const OriginDestExpr LastDestExpr = nullptr,
                 const std::optional<OriginID> LastIssueOriginID = std::nullopt) {
  OriginDestExpr CurrDestExpr = LastDestExpr;
  std::optional<OriginID> IssueOriginID = LastIssueOriginID;
  std::optional<OriginID> CurrOrigin = std::nullopt;
  llvm::ArrayRef<const Fact *> Facts = Context.FactMgr.getFacts(Block);
  bool NeedSearchOriginDestWithoutLoan = false;

  const auto TryInsertAssignmentList = [&](const OriginFlowFact *OFF) {
    if (NeedSearchOriginDestWithoutLoan) {
      if (const MemberExpr *DestMemberExpr =
              dyn_cast_or_null<const MemberExpr *>(
                  getLHSDeclOrExpr(Context, OFF))) {

        CurrDestExpr = DestMemberExpr;
        NeedSearchOriginDestWithoutLoan = false;
      }
    }
    if (OFF->getDestOriginID() == *TargetOID &&
        Context.LoanPropagation.getLoans(OFF->getSrcOriginID(), OFF)
            .contains(EndLoanID)) {
      if (!CurrDestExpr) {
        OriginDestExpr DestExpr = getLHSDeclOrExpr(Context, OFF);
        const ValueDecl *DestValueDecl =
            dyn_cast_or_null<const ValueDecl *>(DestExpr);
        if (DestValueDecl)
          CurrOrigin = *TargetOID;

        if (llvm::isa_and_nonnull<FieldDecl>(DestValueDecl))
          NeedSearchOriginDestWithoutLoan = true;
        else
          CurrDestExpr = DestExpr;
      } else {
        const Expr *CurrSrcExpr = getRHSExpr(Context, OFF);
        if (CurrSrcExpr) {
          AssignmentList.push_back({CurrDestExpr, CurrSrcExpr});
          CurrDestExpr = nullptr;
          CurrOrigin = std::nullopt;
        }
      }
      *TargetOID = OFF->getSrcOriginID();
    }
  };

  for (const Fact *F : llvm::reverse(Facts)) {
    if (const auto *OFF = F->getAs<OriginFlowFact>()) {
      if (IssueOriginID && OFF->getDestOriginID() == IssueOriginID.value())
        return {true, CurrDestExpr, std::nullopt};
      TryInsertAssignmentList(OFF);
    } else if (const auto *IF = F->getAs<IssueFact>()) {
      if (IF->getLoanID() == EndLoanID)
        IssueOriginID = IF->getOriginID();
    } else if (const auto *UF = F->getAs<UseFact>()) {
      if (CurrOrigin) {
        const DeclRefExpr *LHSExpr = getLHSExpr(UF, CurrOrigin.value());
        if (LHSExpr)
          CurrDestExpr = LHSExpr;
      }
    }
  }

  return {false, CurrDestExpr, IssueOriginID};
}

void getAliasListInMultiBlock(
    const AssignmentQueryContext &Context,
    llvm::SmallVectorImpl<AssignmentPair> &AssignmentList,
    const CFGBlock *StartBlock, const LoanID EndLoanID, const OriginID StartOID) {
  std::queue<AssignmentSearchContext> PendingBlocks;
  std::optional<AssignmentPair> FinalAssignment = std::nullopt;
  llvm::SmallSet<AssignmentSearchToken, 32> VistedBlocks;
  llvm::DenseMap<AssignmentPair, AssignmentPair> VistedAssignmentExprs;

  const auto AliasStmtFilter = [&VistedAssignmentExprs,
                                &AssignmentList](const AssignmentPair EndAssignment) {
    AssignmentPair CurrAssignment = EndAssignment;
    while (true) {
      AssignmentList.push_back(CurrAssignment);
      const auto NextAssignment = VistedAssignmentExprs.find(CurrAssignment);
      if (NextAssignment == VistedAssignmentExprs.end())
          break;
      CurrAssignment = NextAssignment->second;
    }
  };

  AssignmentSearchToken StartToken = {StartBlock, StartOID};
  AssignmentSearchContext StartContext = {StartToken, nullptr, std::nullopt, std::nullopt};
  PendingBlocks.push(StartContext);

  while (!PendingBlocks.empty()) {
    const AssignmentSearchContext CurrContext = PendingBlocks.front();
    PendingBlocks.pop();

    std::optional<AssignmentPair> EndAssignment = std::nullopt;
    llvm::SmallVector<AssignmentPair> BlockAliasList;
    OriginID CurrOID = CurrContext.CurrToken.OID;

    const AliasAssignmentSearchResult Result =
        getAliasListCore(Context, BlockAliasList, CurrContext.CurrToken.Block, EndLoanID,
                         &CurrOID, CurrContext.LastDestDeclOrExpr, CurrContext.LastOriginID);

    if (!BlockAliasList.empty()) {
      for (size_t i = 0; i < BlockAliasList.size() - 1; ++i)
        VistedAssignmentExprs.insert(
            {BlockAliasList[i + 1], BlockAliasList[i]});

      if (CurrContext.LastEndAssignment)
        VistedAssignmentExprs.insert({BlockAliasList[0], CurrContext.LastEndAssignment.value()});

      EndAssignment = BlockAliasList.back();
      FinalAssignment = BlockAliasList.back();
    } else {
      EndAssignment = CurrContext.LastEndAssignment;
    }

    // TODO: The number of CFGBlocks is limited to 32 to minmize performance
    // impact. Note that is not a magic number derived from extensive
    // engineering practice. If this limit proves unnecessary or overly
    // restrictive, the boundary conditions should be adjusted.
    // https://github.com/llvm/llvm-project/pull/188467#discussion_r3050068533
    if (Result.SearchComplete || VistedBlocks.size() >= 32)
      break;

    for (const CFGBlock *NextBlock : CurrContext.CurrToken.Block->preds()) {
      AssignmentSearchToken NextToken = {NextBlock, CurrOID};
      if (NextBlock && VistedBlocks.insert(NextToken).second) {
        AssignmentSearchContext NextContext = {NextToken, Result.LastDestDeclOrExpr, EndAssignment, Result.IssueOriginID};
        PendingBlocks.push(NextContext);
      }
    }
  }

  if (FinalAssignment)
    AliasStmtFilter(FinalAssignment.value());
}

void formatRHSValueDeclForSema(const ValueDecl *TargetValue,
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

void formatLoanEntityForSema(LoanEntity IssueEntity,
                             llvm::SmallVectorImpl<char> &IssueMsg) {
  llvm::StringRef Temp = "the temporary";

  if (const auto *IssueExpr = dyn_cast<const Expr *>(IssueEntity)) {
    if (const auto *IDeclExpr =
            dyn_cast<DeclRefExpr>(IssueExpr->IgnoreParenCasts())) {
      formatRHSValueDeclForSema(IDeclExpr->getDecl(), IssueMsg);
      return;
    }
  }

  if (const auto *IssueParmDecl = dyn_cast<const ParmVarDecl *>(IssueEntity))
    formatRHSValueDeclForSema(IssueParmDecl, IssueMsg);
  else if (const auto *IssueCXXMD =
               dyn_cast<const CXXMethodDecl *>(IssueEntity))
    formatRHSValueDeclForSema(IssueCXXMD, IssueMsg);
  else
    IssueMsg.append(Temp.begin(), Temp.end());
}

/// Recursively parse the `SrcExpr` and issue corresponding warnings based on
/// its type.
///
/// Most expressions handled here are obtained from `getPureSrcExpr`, but we
/// must account for additional expression types encountered during recursion.
///
/// Certain implicit expressions, such as `CXXBindTemporaryExpr` and
/// `CXXConstructExpr`, are allowed and should be skipped without performing
/// `Expr->getExprLoc().isValid()` validation.
void formatSrcExprForSema(
    const Expr *SrcExpr,
    llvm::SmallVectorImpl<ExprPrintingResult> &SrcMsgList) {
  if (!SrcExpr)
    return;
  const Expr *PureExpr = SrcExpr->IgnoreParenCasts();

  const auto AppendSubExprIfNeeded = [&SrcMsgList](const Expr *SubExpr) {
    if (SubExpr && !llvm::isa<DeclRefExpr>(SubExpr->IgnoreParenCasts())) {
      formatSrcExprForSema(SubExpr, SrcMsgList);
    }
  };

  if (const auto *SrcCXXOpCallExpr =
          dyn_cast_or_null<CXXOperatorCallExpr>(PureExpr);
      SrcCXXOpCallExpr && SrcCXXOpCallExpr->getExprLoc().isValid()) {
    SrcMsgList.push_back({"expression", SrcCXXOpCallExpr->getArg(0)});
  } else if (const auto *SrcDeclRefExpr =
                 dyn_cast_or_null<DeclRefExpr>(PureExpr);
             SrcDeclRefExpr && SrcDeclRefExpr->getExprLoc().isValid()) {
    SrcMsgList.push_back({"", SrcDeclRefExpr});
  } else if (const auto *SrcCXXMemberCallExpr =
                 dyn_cast_or_null<CXXMemberCallExpr>(PureExpr);
             SrcCXXMemberCallExpr &&
             SrcCXXMemberCallExpr->getExprLoc().isValid()) {
    if (SrcCXXMemberCallExpr->getCallee()->getExprLoc().isValid())
      SrcMsgList.push_back({"function call result", SrcCXXMemberCallExpr});
    AppendSubExprIfNeeded(SrcCXXMemberCallExpr->getImplicitObjectArgument());
  } else if (const auto *SrcCallExpr = dyn_cast_or_null<CallExpr>(PureExpr);
             SrcCallExpr && SrcCallExpr->getExprLoc().isValid()) {
    if (SrcCallExpr->getCallee()->getExprLoc().isValid())
      SrcMsgList.push_back({"function call result", SrcCallExpr});
    AppendSubExprIfNeeded(SrcCallExpr->getCallee());
  } else if (const auto *SrcMemberExpr = dyn_cast_or_null<MemberExpr>(PureExpr);
             SrcMemberExpr && SrcMemberExpr->getExprLoc().isValid() &&
             !SrcMemberExpr->getMemberDecl()->isImplicit()) {
    SrcMsgList.push_back({"member access", SrcMemberExpr});
    AppendSubExprIfNeeded(SrcMemberExpr->getBase());
  } else if (const auto *SrcCXXConExpr =
                 dyn_cast_or_null<CXXConstructExpr>(PureExpr);
             SrcCXXConExpr && SrcCXXConExpr->getNumArgs() > 0) {
    AppendSubExprIfNeeded(SrcCXXConExpr->getArg(0));
  } else if (const auto *SrcCXXBindTempExpr =
                 dyn_cast_or_null<CXXBindTemporaryExpr>(PureExpr)) {
    AppendSubExprIfNeeded(SrcCXXBindTempExpr->getSubExpr());
  } else if (const auto *SrcArraySubExpr =
                 dyn_cast_or_null<ArraySubscriptExpr>(PureExpr);
             SrcArraySubExpr && SrcArraySubExpr->getExprLoc().isValid()) {
    SrcMsgList.push_back({"array subscript access", SrcArraySubExpr});
    AppendSubExprIfNeeded(SrcArraySubExpr->getBase());
  } else if (const auto *SrcUnaryOpExpr =
                 dyn_cast_or_null<UnaryOperator>(PureExpr)) {
    AppendSubExprIfNeeded(SrcUnaryOpExpr->getSubExpr());
  }
}
} // namespace clang::lifetimes

namespace clang::lifetimes::internal {

void getAliasList(const AssignmentQueryContext &Context,
                  llvm::SmallVectorImpl<AssignmentPair> &AssignmentList,
                  const Fact *CausingFact, const LoanID End,
                  const CFGBlock *StartBlock, const Expr *IssueExpr) {
  llvm::SmallVector<OriginID, 4> TargetOIDList;

  if (const auto *UF = dyn_cast<UseFact>(CausingFact))
    for (const OriginList *Cur = UF->getUsedOrigins(); Cur;
         Cur = Cur->peelOuterOrigin())
      TargetOIDList.push_back(Cur->getOuterOriginID());
  else if (const auto *RetEscapeF = dyn_cast<ReturnEscapeFact>(CausingFact))
    TargetOIDList.push_back(RetEscapeF->getEscapedOriginID());
  else if (const auto *FieldEscapeF = dyn_cast<FieldEscapeFact>(CausingFact))
    TargetOIDList.push_back(FieldEscapeF->getEscapedOriginID());
  else if (const auto *GlobalEscapeF = dyn_cast<GlobalEscapeFact>(CausingFact))
    TargetOIDList.push_back(GlobalEscapeF->getEscapedOriginID());
  else
    llvm_unreachable("Without a corresponding Fact handler, assignment history "
                     "traceback will fail.");

  const CFGBlock *EndBlock =
      IssueExpr ? Context.ADC.getCFGStmtMap()->getBlock(IssueExpr) : nullptr;

  for (OriginID TargetOID : TargetOIDList) {
    if (StartBlock == EndBlock) {
      getAliasListCore(Context, AssignmentList, StartBlock, End, &TargetOID);
      std::reverse(AssignmentList.begin(), AssignmentList.end());
    } else {
      getAliasListInMultiBlock(Context, AssignmentList, StartBlock, End,
                               TargetOID);
    }
  }
}
} // namespace clang::lifetimes::internal
