//=- ReachableCodePathInsensitive.cpp ---------------------------*- C++ --*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a flow-sensitive, path-insensitive analysis of
// determining reachable blocks within a CFG.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Analysis/Analyses/ReachableCode.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Basic/SourceManager.h"

using namespace clang;

namespace {
class DeadCodeScan {
  llvm::BitVector Visited;
  llvm::BitVector &Reachable;
  llvm::SmallVector<const CFGBlock *, 10> WorkList;
  
  typedef llvm::SmallVector<std::pair<const CFGBlock *, const Stmt *>, 12>
      DeferredLocsTy;
  
  DeferredLocsTy DeferredLocs;
  
public:
  DeadCodeScan(llvm::BitVector &reachable)
    : Visited(reachable.size()),
      Reachable(reachable) {}
  
  void enqueue(const CFGBlock *block);  
  unsigned scanBackwards(const CFGBlock *Start,
                         clang::reachable_code::Callback &CB);
  
  bool isDeadCodeRoot(const CFGBlock *Block);
  
  const Stmt *findDeadCode(const CFGBlock *Block);
  
  void reportDeadCode(const Stmt *S,
                      clang::reachable_code::Callback &CB);
};
}

void DeadCodeScan::enqueue(const CFGBlock *block) {  
  unsigned blockID = block->getBlockID();
  if (Reachable[blockID] || Visited[blockID])
    return;
  Visited[blockID] = true;
  WorkList.push_back(block);
}

bool DeadCodeScan::isDeadCodeRoot(const clang::CFGBlock *Block) {
  bool isDeadRoot = true;
  
  for (CFGBlock::const_pred_iterator I = Block->pred_begin(),
        E = Block->pred_end(); I != E; ++I) {
    if (const CFGBlock *PredBlock = *I) {
      unsigned blockID = PredBlock->getBlockID();
      if (Visited[blockID]) {
        isDeadRoot = false;
        continue;
      }
      if (!Reachable[blockID]) {
        isDeadRoot = false;
        Visited[blockID] = true;
        WorkList.push_back(PredBlock);
        continue;
      }
    }
  }
  
  return isDeadRoot;
}

static bool isValidDeadStmt(const Stmt *S) {
  if (S->getLocStart().isInvalid())
    return false;
  if (const BinaryOperator *BO = dyn_cast<BinaryOperator>(S))
    return BO->getOpcode() != BO_Comma;
  return true;
}

const Stmt *DeadCodeScan::findDeadCode(const clang::CFGBlock *Block) {
  for (CFGBlock::const_iterator I = Block->begin(), E = Block->end(); I!=E; ++I)
    if (const CFGStmt *CS = I->getAs<CFGStmt>()) {
      const Stmt *S = CS->getStmt();
      if (isValidDeadStmt(S))
        return S;
    }
  
  if (CFGTerminator T = Block->getTerminator()) {
    const Stmt *S = T.getStmt();
    if (isValidDeadStmt(S))
      return S;    
  }

  return 0;
}

static int SrcCmp(const void *p1, const void *p2) {
  return
    ((const std::pair<const CFGBlock *, const Stmt *>*) p2)->second->getLocStart() <
    ((const std::pair<const CFGBlock *, const Stmt *>*) p1)->second->getLocStart();
}

unsigned DeadCodeScan::scanBackwards(const clang::CFGBlock *Start,
                                     clang::reachable_code::Callback &CB) {

  unsigned count = 0;
  enqueue(Start);
  
  while (!WorkList.empty()) {
    const CFGBlock *Block = WorkList.pop_back_val();

    // It is possible that this block has been marked reachable after
    // it was enqueued.
    if (Reachable[Block->getBlockID()])
      continue;

    // Look for any dead code within the block.
    const Stmt *S = findDeadCode(Block);
    
    if (!S) {
      // No dead code.  Possibly an empty block.  Look at dead predecessors.
      for (CFGBlock::const_pred_iterator I = Block->pred_begin(),
           E = Block->pred_end(); I != E; ++I) {
        if (const CFGBlock *predBlock = *I)
          enqueue(predBlock);
      }
      continue;
    }
    
    // Specially handle macro-expanded code.
    if (S->getLocStart().isMacroID()) {
      count += clang::reachable_code::ScanReachableFromBlock(Block, Reachable);
      continue;
    }

    if (isDeadCodeRoot(Block)) {
      reportDeadCode(S, CB);
      count += clang::reachable_code::ScanReachableFromBlock(Block, Reachable);
    }
    else {
      // Record this statement as the possibly best location in a
      // strongly-connected component of dead code for emitting a
      // warning.
      DeferredLocs.push_back(std::make_pair(Block, S));
    }
  }

  // If we didn't find a dead root, then report the dead code with the
  // earliest location.
  if (!DeferredLocs.empty()) {
    llvm::array_pod_sort(DeferredLocs.begin(), DeferredLocs.end(), SrcCmp);
    for (DeferredLocsTy::iterator I = DeferredLocs.begin(),
          E = DeferredLocs.end(); I != E; ++I) {
      const CFGBlock *block = I->first;
      if (Reachable[block->getBlockID()])
        continue;
      reportDeadCode(I->second, CB);
      count += clang::reachable_code::ScanReachableFromBlock(block, Reachable);
    }
  }
    
  return count;
}

static SourceLocation GetUnreachableLoc(const Stmt *S,
                                        SourceRange &R1,
                                        SourceRange &R2) {
  R1 = R2 = SourceRange();

  if (const Expr *Ex = dyn_cast<Expr>(S))
    S = Ex->IgnoreParenImpCasts();

  switch (S->getStmtClass()) {
    case Expr::BinaryOperatorClass: {
      const BinaryOperator *BO = cast<BinaryOperator>(S);
      return BO->getOperatorLoc();
    }
    case Expr::UnaryOperatorClass: {
      const UnaryOperator *UO = cast<UnaryOperator>(S);
      R1 = UO->getSubExpr()->getSourceRange();
      return UO->getOperatorLoc();
    }
    case Expr::CompoundAssignOperatorClass: {
      const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(S);
      R1 = CAO->getLHS()->getSourceRange();
      R2 = CAO->getRHS()->getSourceRange();
      return CAO->getOperatorLoc();
    }
    case Expr::BinaryConditionalOperatorClass:
    case Expr::ConditionalOperatorClass: {
      const AbstractConditionalOperator *CO =
        cast<AbstractConditionalOperator>(S);
      return CO->getQuestionLoc();
    }
    case Expr::MemberExprClass: {
      const MemberExpr *ME = cast<MemberExpr>(S);
      R1 = ME->getSourceRange();
      return ME->getMemberLoc();
    }
    case Expr::ArraySubscriptExprClass: {
      const ArraySubscriptExpr *ASE = cast<ArraySubscriptExpr>(S);
      R1 = ASE->getLHS()->getSourceRange();
      R2 = ASE->getRHS()->getSourceRange();
      return ASE->getRBracketLoc();
    }
    case Expr::CStyleCastExprClass: {
      const CStyleCastExpr *CSC = cast<CStyleCastExpr>(S);
      R1 = CSC->getSubExpr()->getSourceRange();
      return CSC->getLParenLoc();
    }
    case Expr::CXXFunctionalCastExprClass: {
      const CXXFunctionalCastExpr *CE = cast <CXXFunctionalCastExpr>(S);
      R1 = CE->getSubExpr()->getSourceRange();
      return CE->getTypeBeginLoc();
    }
    case Stmt::CXXTryStmtClass: {
      return cast<CXXTryStmt>(S)->getHandler(0)->getCatchLoc();
    }
    case Expr::ObjCBridgedCastExprClass: {
      const ObjCBridgedCastExpr *CSC = cast<ObjCBridgedCastExpr>(S);
      R1 = CSC->getSubExpr()->getSourceRange();
      return CSC->getLParenLoc();
    }
    default: ;
  }
  R1 = S->getSourceRange();
  return S->getLocStart();
}

void DeadCodeScan::reportDeadCode(const Stmt *S,
                                  clang::reachable_code::Callback &CB) {
  SourceRange R1, R2;
  SourceLocation Loc = GetUnreachableLoc(S, R1, R2);
  CB.HandleUnreachable(Loc, R1, R2);
}

namespace clang { namespace reachable_code {

void Callback::anchor() { }  

unsigned ScanReachableFromBlock(const CFGBlock *Start,
                                llvm::BitVector &Reachable) {
  unsigned count = 0;
  
  // Prep work queue
  SmallVector<const CFGBlock*, 32> WL;
  
  // The entry block may have already been marked reachable
  // by the caller.
  if (!Reachable[Start->getBlockID()]) {
    ++count;
    Reachable[Start->getBlockID()] = true;
  }
  
  WL.push_back(Start);
  
  // Find the reachable blocks from 'Start'.
  while (!WL.empty()) {
    const CFGBlock *item = WL.pop_back_val();
    
    // Look at the successors and mark then reachable.
    for (CFGBlock::const_succ_iterator I = item->succ_begin(), 
         E = item->succ_end(); I != E; ++I)
      if (const CFGBlock *B = *I) {
        unsigned blockID = B->getBlockID();
        if (!Reachable[blockID]) {
          Reachable.set(blockID);
          WL.push_back(B);
          ++count;
        }
      }
  }
  return count;
}
  
void FindUnreachableCode(AnalysisDeclContext &AC, Callback &CB) {
  CFG *cfg = AC.getCFG();
  if (!cfg)
    return;

  // Scan for reachable blocks from the entrance of the CFG.  
  // If there are no unreachable blocks, we're done.
  llvm::BitVector reachable(cfg->getNumBlockIDs());
  unsigned numReachable = ScanReachableFromBlock(&cfg->getEntry(), reachable);
  if (numReachable == cfg->getNumBlockIDs())
    return;
  
  // If there aren't explicit EH edges, we should include the 'try' dispatch
  // blocks as roots.
  if (!AC.getCFGBuildOptions().AddEHEdges) {
    for (CFG::try_block_iterator I = cfg->try_blocks_begin(),
         E = cfg->try_blocks_end() ; I != E; ++I) {
      numReachable += ScanReachableFromBlock(*I, reachable);
    }
    if (numReachable == cfg->getNumBlockIDs())
      return;
  }

  // There are some unreachable blocks.  We need to find the root blocks that
  // contain code that should be considered unreachable.  
  for (CFG::iterator I = cfg->begin(), E = cfg->end(); I != E; ++I) {
    const CFGBlock *block = *I;
    // A block may have been marked reachable during this loop.
    if (reachable[block->getBlockID()])
      continue;
    
    DeadCodeScan DS(reachable);
    numReachable += DS.scanBackwards(block, CB);
    
    if (numReachable == cfg->getNumBlockIDs())
      return;
  }
}

}} // end namespace clang::reachable_code
