//==- UninitializedValues.cpp - Find Uninitialized Values -------*- C++ --*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements uninitialized values analysis for source-level CFGs.
//
//===----------------------------------------------------------------------===//

#include <utility>
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/PackedVector.h"
#include "llvm/ADT/DenseMap.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/Visitors/CFGRecStmtDeclVisitor.h"
#include "clang/Analysis/Analyses/UninitializedValues.h"
#include "clang/Analysis/DomainSpecific/ObjCNoReturn.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace clang;

#define DEBUG_LOGGING 0

static bool isTrackedVar(const VarDecl *vd, const DeclContext *dc) {
  if (vd->isLocalVarDecl() && !vd->hasGlobalStorage() &&
      !vd->isExceptionVariable() &&
      vd->getDeclContext() == dc) {
    QualType ty = vd->getType();
    return ty->isScalarType() || ty->isVectorType();
  }
  return false;
}

//------------------------------------------------------------------------====//
// DeclToIndex: a mapping from Decls we track to value indices.
//====------------------------------------------------------------------------//

namespace {
class DeclToIndex {
  llvm::DenseMap<const VarDecl *, unsigned> map;
public:
  DeclToIndex() {}
  
  /// Compute the actual mapping from declarations to bits.
  void computeMap(const DeclContext &dc);
  
  /// Return the number of declarations in the map.
  unsigned size() const { return map.size(); }
  
  /// Returns the bit vector index for a given declaration.
  llvm::Optional<unsigned> getValueIndex(const VarDecl *d) const;
};
}

void DeclToIndex::computeMap(const DeclContext &dc) {
  unsigned count = 0;
  DeclContext::specific_decl_iterator<VarDecl> I(dc.decls_begin()),
                                               E(dc.decls_end());
  for ( ; I != E; ++I) {
    const VarDecl *vd = *I;
    if (isTrackedVar(vd, &dc))
      map[vd] = count++;
  }
}

llvm::Optional<unsigned> DeclToIndex::getValueIndex(const VarDecl *d) const {
  llvm::DenseMap<const VarDecl *, unsigned>::const_iterator I = map.find(d);
  if (I == map.end())
    return llvm::Optional<unsigned>();
  return I->second;
}

//------------------------------------------------------------------------====//
// CFGBlockValues: dataflow values for CFG blocks.
//====------------------------------------------------------------------------//

// These values are defined in such a way that a merge can be done using
// a bitwise OR.
enum Value { Unknown = 0x0,         /* 00 */
             Initialized = 0x1,     /* 01 */
             Uninitialized = 0x2,   /* 10 */
             MayUninitialized = 0x3 /* 11 */ };

static bool isUninitialized(const Value v) {
  return v >= Uninitialized;
}
static bool isAlwaysUninit(const Value v) {
  return v == Uninitialized;
}

namespace {

typedef llvm::PackedVector<Value, 2, llvm::SmallBitVector> ValueVector;

class CFGBlockValues {
  const CFG &cfg;
  SmallVector<ValueVector, 8> vals;
  ValueVector scratch;
  DeclToIndex declToIndex;
public:
  CFGBlockValues(const CFG &cfg);

  unsigned getNumEntries() const { return declToIndex.size(); }
  
  void computeSetOfDeclarations(const DeclContext &dc);  
  ValueVector &getValueVector(const CFGBlock *block) {
    return vals[block->getBlockID()];
  }

  void setAllScratchValues(Value V);
  void mergeIntoScratch(ValueVector const &source, bool isFirst);
  bool updateValueVectorWithScratch(const CFGBlock *block);
  
  bool hasNoDeclarations() const {
    return declToIndex.size() == 0;
  }

  void resetScratch();
  
  ValueVector::reference operator[](const VarDecl *vd);

  Value getValue(const CFGBlock *block, const CFGBlock *dstBlock,
                 const VarDecl *vd) {
    const llvm::Optional<unsigned> &idx = declToIndex.getValueIndex(vd);
    assert(idx.hasValue());
    return getValueVector(block)[idx.getValue()];
  }
};  
} // end anonymous namespace

CFGBlockValues::CFGBlockValues(const CFG &c) : cfg(c), vals(0) {}

void CFGBlockValues::computeSetOfDeclarations(const DeclContext &dc) {
  declToIndex.computeMap(dc);
  unsigned decls = declToIndex.size();
  scratch.resize(decls);
  unsigned n = cfg.getNumBlockIDs();
  if (!n)
    return;
  vals.resize(n);
  for (unsigned i = 0; i < n; ++i)
    vals[i].resize(decls);
}

#if DEBUG_LOGGING
static void printVector(const CFGBlock *block, ValueVector &bv,
                        unsigned num) {
  llvm::errs() << block->getBlockID() << " :";
  for (unsigned i = 0; i < bv.size(); ++i) {
    llvm::errs() << ' ' << bv[i];
  }
  llvm::errs() << " : " << num << '\n';
}
#endif

void CFGBlockValues::setAllScratchValues(Value V) {
  for (unsigned I = 0, E = scratch.size(); I != E; ++I)
    scratch[I] = V;
}

void CFGBlockValues::mergeIntoScratch(ValueVector const &source,
                                      bool isFirst) {
  if (isFirst)
    scratch = source;
  else
    scratch |= source;
}

bool CFGBlockValues::updateValueVectorWithScratch(const CFGBlock *block) {
  ValueVector &dst = getValueVector(block);
  bool changed = (dst != scratch);
  if (changed)
    dst = scratch;
#if DEBUG_LOGGING
  printVector(block, scratch, 0);
#endif
  return changed;
}

void CFGBlockValues::resetScratch() {
  scratch.reset();
}

ValueVector::reference CFGBlockValues::operator[](const VarDecl *vd) {
  const llvm::Optional<unsigned> &idx = declToIndex.getValueIndex(vd);
  assert(idx.hasValue());
  return scratch[idx.getValue()];
}

//------------------------------------------------------------------------====//
// Worklist: worklist for dataflow analysis.
//====------------------------------------------------------------------------//

namespace {
class DataflowWorklist {
  SmallVector<const CFGBlock *, 20> worklist;
  llvm::BitVector enqueuedBlocks;
public:
  DataflowWorklist(const CFG &cfg) : enqueuedBlocks(cfg.getNumBlockIDs()) {}
  
  void enqueueSuccessors(const CFGBlock *block);
  const CFGBlock *dequeue();
};
}

void DataflowWorklist::enqueueSuccessors(const clang::CFGBlock *block) {
  unsigned OldWorklistSize = worklist.size();
  for (CFGBlock::const_succ_iterator I = block->succ_begin(),
       E = block->succ_end(); I != E; ++I) {
    const CFGBlock *Successor = *I;
    if (!Successor || enqueuedBlocks[Successor->getBlockID()])
      continue;
    worklist.push_back(Successor);
    enqueuedBlocks[Successor->getBlockID()] = true;
  }
  if (OldWorklistSize == 0 || OldWorklistSize == worklist.size())
    return;

  // Rotate the newly added blocks to the start of the worklist so that it forms
  // a proper queue when we pop off the end of the worklist.
  std::rotate(worklist.begin(), worklist.begin() + OldWorklistSize,
              worklist.end());
}

const CFGBlock *DataflowWorklist::dequeue() {
  if (worklist.empty())
    return 0;
  const CFGBlock *b = worklist.back();
  worklist.pop_back();
  enqueuedBlocks[b->getBlockID()] = false;
  return b;
}

//------------------------------------------------------------------------====//
// Classification of DeclRefExprs as use or initialization.
//====------------------------------------------------------------------------//

namespace {
class FindVarResult {
  const VarDecl *vd;
  const DeclRefExpr *dr;
public:
  FindVarResult(const VarDecl *vd, const DeclRefExpr *dr) : vd(vd), dr(dr) {}

  const DeclRefExpr *getDeclRefExpr() const { return dr; }
  const VarDecl *getDecl() const { return vd; }
};

static const Expr *stripCasts(ASTContext &C, const Expr *Ex) {
  while (Ex) {
    Ex = Ex->IgnoreParenNoopCasts(C);
    if (const CastExpr *CE = dyn_cast<CastExpr>(Ex)) {
      if (CE->getCastKind() == CK_LValueBitCast) {
        Ex = CE->getSubExpr();
        continue;
      }
    }
    break;
  }
  return Ex;
}

/// If E is an expression comprising a reference to a single variable, find that
/// variable.
static FindVarResult findVar(const Expr *E, const DeclContext *DC) {
  if (const DeclRefExpr *DRE =
        dyn_cast<DeclRefExpr>(stripCasts(DC->getParentASTContext(), E)))
    if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      if (isTrackedVar(VD, DC))
        return FindVarResult(VD, DRE);
  return FindVarResult(0, 0);
}

/// \brief Classify each DeclRefExpr as an initialization or a use. Any
/// DeclRefExpr which isn't explicitly classified will be assumed to have
/// escaped the analysis and will be treated as an initialization.
class ClassifyRefs : public StmtVisitor<ClassifyRefs> {
public:
  enum Class {
    Init,
    Use,
    SelfInit,
    Ignore
  };

private:
  const DeclContext *DC;
  llvm::DenseMap<const DeclRefExpr*, Class> Classification;

  bool isTrackedVar(const VarDecl *VD) const {
    return ::isTrackedVar(VD, DC);
  }

  void classify(const Expr *E, Class C);

public:
  ClassifyRefs(AnalysisDeclContext &AC) : DC(cast<DeclContext>(AC.getDecl())) {}

  void VisitDeclStmt(DeclStmt *DS);
  void VisitUnaryOperator(UnaryOperator *UO);
  void VisitBinaryOperator(BinaryOperator *BO);
  void VisitCallExpr(CallExpr *CE);
  void VisitCastExpr(CastExpr *CE);

  void operator()(Stmt *S) { Visit(S); }

  Class get(const DeclRefExpr *DRE) const {
    llvm::DenseMap<const DeclRefExpr*, Class>::const_iterator I
        = Classification.find(DRE);
    if (I != Classification.end())
      return I->second;

    const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());
    if (!VD || !isTrackedVar(VD))
      return Ignore;

    return Init;
  }
};
}

static const DeclRefExpr *getSelfInitExpr(VarDecl *VD) {
  if (Expr *Init = VD->getInit()) {
    const DeclRefExpr *DRE
      = dyn_cast<DeclRefExpr>(stripCasts(VD->getASTContext(), Init));
    if (DRE && DRE->getDecl() == VD)
      return DRE;
  }
  return 0;
}

void ClassifyRefs::classify(const Expr *E, Class C) {
  FindVarResult Var = findVar(E, DC);
  if (const DeclRefExpr *DRE = Var.getDeclRefExpr())
    Classification[DRE] = std::max(Classification[DRE], C);
}

void ClassifyRefs::VisitDeclStmt(DeclStmt *DS) {
  for (DeclStmt::decl_iterator DI = DS->decl_begin(), DE = DS->decl_end();
       DI != DE; ++DI) {
    VarDecl *VD = dyn_cast<VarDecl>(*DI);
    if (VD && isTrackedVar(VD))
      if (const DeclRefExpr *DRE = getSelfInitExpr(VD))
        Classification[DRE] = SelfInit;
  }
}

void ClassifyRefs::VisitBinaryOperator(BinaryOperator *BO) {
  // Ignore the evaluation of a DeclRefExpr on the LHS of an assignment. If this
  // is not a compound-assignment, we will treat it as initializing the variable
  // when TransferFunctions visits it. A compound-assignment does not affect
  // whether a variable is uninitialized, and there's no point counting it as a
  // use.
  if (BO->isCompoundAssignmentOp())
    classify(BO->getLHS(), Use);
  else if (BO->getOpcode() == BO_Assign)
    classify(BO->getLHS(), Ignore);
}

void ClassifyRefs::VisitUnaryOperator(UnaryOperator *UO) {
  // Increment and decrement are uses despite there being no lvalue-to-rvalue
  // conversion.
  if (UO->isIncrementDecrementOp())
    classify(UO->getSubExpr(), Use);
}

void ClassifyRefs::VisitCallExpr(CallExpr *CE) {
  // If a value is passed by const reference to a function, we should not assume
  // that it is initialized by the call, and we conservatively do not assume
  // that it is used.
  for (CallExpr::arg_iterator I = CE->arg_begin(), E = CE->arg_end();
       I != E; ++I)
    if ((*I)->getType().isConstQualified() && (*I)->isGLValue())
      classify(*I, Ignore);
}

void ClassifyRefs::VisitCastExpr(CastExpr *CE) {
  if (CE->getCastKind() == CK_LValueToRValue)
    classify(CE->getSubExpr(), Use);
  else if (CStyleCastExpr *CSE = dyn_cast<CStyleCastExpr>(CE)) {
    if (CSE->getType()->isVoidType()) {
      // Squelch any detected load of an uninitialized value if
      // we cast it to void.
      // e.g. (void) x;
      classify(CSE->getSubExpr(), Ignore);
    }
  }
}

//------------------------------------------------------------------------====//
// Transfer function for uninitialized values analysis.
//====------------------------------------------------------------------------//

namespace {
class TransferFunctions : public StmtVisitor<TransferFunctions> {
  CFGBlockValues &vals;
  const CFG &cfg;
  const CFGBlock *block;
  AnalysisDeclContext &ac;
  const ClassifyRefs &classification;
  ObjCNoReturn objCNoRet;
  UninitVariablesHandler *handler;

public:
  TransferFunctions(CFGBlockValues &vals, const CFG &cfg,
                    const CFGBlock *block, AnalysisDeclContext &ac,
                    const ClassifyRefs &classification,
                    UninitVariablesHandler *handler)
    : vals(vals), cfg(cfg), block(block), ac(ac),
      classification(classification), objCNoRet(ac.getASTContext()),
      handler(handler) {}

  void reportUse(const Expr *ex, const VarDecl *vd);

  void VisitBinaryOperator(BinaryOperator *bo);
  void VisitBlockExpr(BlockExpr *be);
  void VisitCallExpr(CallExpr *ce);
  void VisitDeclRefExpr(DeclRefExpr *dr);
  void VisitDeclStmt(DeclStmt *ds);
  void VisitObjCForCollectionStmt(ObjCForCollectionStmt *FS);
  void VisitObjCMessageExpr(ObjCMessageExpr *ME);

  bool isTrackedVar(const VarDecl *vd) {
    return ::isTrackedVar(vd, cast<DeclContext>(ac.getDecl()));
  }

  FindVarResult findVar(const Expr *ex) {
    return ::findVar(ex, cast<DeclContext>(ac.getDecl()));
  }

  UninitUse getUninitUse(const Expr *ex, const VarDecl *vd, Value v) {
    UninitUse Use(ex, isAlwaysUninit(v));

    assert(isUninitialized(v));
    if (Use.getKind() == UninitUse::Always)
      return Use;

    // If an edge which leads unconditionally to this use did not initialize
    // the variable, we can say something stronger than 'may be uninitialized':
    // we can say 'either it's used uninitialized or you have dead code'.
    //
    // We track the number of successors of a node which have been visited, and
    // visit a node once we have visited all of its successors. Only edges where
    // the variable might still be uninitialized are followed. Since a variable
    // can't transfer from being initialized to being uninitialized, this will
    // trace out the subgraph which inevitably leads to the use and does not
    // initialize the variable. We do not want to skip past loops, since their
    // non-termination might be correlated with the initialization condition.
    //
    // For example:
    //
    //         void f(bool a, bool b) {
    // block1:   int n;
    //           if (a) {
    // block2:     if (b)
    // block3:       n = 1;
    // block4:   } else if (b) {
    // block5:     while (!a) {
    // block6:       do_work(&a);
    //               n = 2;
    //             }
    //           }
    // block7:   if (a)
    // block8:     g();
    // block9:   return n;
    //         }
    //
    // Starting from the maybe-uninitialized use in block 9:
    //  * Block 7 is not visited because we have only visited one of its two
    //    successors.
    //  * Block 8 is visited because we've visited its only successor.
    // From block 8:
    //  * Block 7 is visited because we've now visited both of its successors.
    // From block 7:
    //  * Blocks 1, 2, 4, 5, and 6 are not visited because we didn't visit all
    //    of their successors (we didn't visit 4, 3, 5, 6, and 5, respectively).
    //  * Block 3 is not visited because it initializes 'n'.
    // Now the algorithm terminates, having visited blocks 7 and 8, and having
    // found the frontier is blocks 2, 4, and 5.
    //
    // 'n' is definitely uninitialized for two edges into block 7 (from blocks 2
    // and 4), so we report that any time either of those edges is taken (in
    // each case when 'b == false'), 'n' is used uninitialized.
    llvm::SmallVector<const CFGBlock*, 32> Queue;
    llvm::SmallVector<unsigned, 32> SuccsVisited(cfg.getNumBlockIDs(), 0);
    Queue.push_back(block);
    // Specify that we've already visited all successors of the starting block.
    // This has the dual purpose of ensuring we never add it to the queue, and
    // of marking it as not being a candidate element of the frontier.
    SuccsVisited[block->getBlockID()] = block->succ_size();
    while (!Queue.empty()) {
      const CFGBlock *B = Queue.back();
      Queue.pop_back();
      for (CFGBlock::const_pred_iterator I = B->pred_begin(), E = B->pred_end();
           I != E; ++I) {
        const CFGBlock *Pred = *I;
        if (vals.getValue(Pred, B, vd) == Initialized)
          // This block initializes the variable.
          continue;

        unsigned &SV = SuccsVisited[Pred->getBlockID()];
        if (!SV) {
          // When visiting the first successor of a block, mark all NULL
          // successors as having been visited.
          for (CFGBlock::const_succ_iterator SI = Pred->succ_begin(),
                                             SE = Pred->succ_end();
               SI != SE; ++SI)
            if (!*SI)
              ++SV;
        }

        if (++SV == Pred->succ_size())
          // All paths from this block lead to the use and don't initialize the
          // variable.
          Queue.push_back(Pred);
      }
    }

    // Scan the frontier, looking for blocks where the variable was
    // uninitialized.
    for (CFG::const_iterator BI = cfg.begin(), BE = cfg.end(); BI != BE; ++BI) {
      const CFGBlock *Block = *BI;
      unsigned BlockID = Block->getBlockID();
      const Stmt *Term = Block->getTerminator();
      if (SuccsVisited[BlockID] && SuccsVisited[BlockID] < Block->succ_size() &&
          Term) {
        // This block inevitably leads to the use. If we have an edge from here
        // to a post-dominator block, and the variable is uninitialized on that
        // edge, we have found a bug.
        for (CFGBlock::const_succ_iterator I = Block->succ_begin(),
             E = Block->succ_end(); I != E; ++I) {
          const CFGBlock *Succ = *I;
          if (Succ && SuccsVisited[Succ->getBlockID()] >= Succ->succ_size() &&
              vals.getValue(Block, Succ, vd) == Uninitialized) {
            // Switch cases are a special case: report the label to the caller
            // as the 'terminator', not the switch statement itself. Suppress
            // situations where no label matched: we can't be sure that's
            // possible.
            if (isa<SwitchStmt>(Term)) {
              const Stmt *Label = Succ->getLabel();
              if (!Label || !isa<SwitchCase>(Label))
                // Might not be possible.
                continue;
              UninitUse::Branch Branch;
              Branch.Terminator = Label;
              Branch.Output = 0; // Ignored.
              Use.addUninitBranch(Branch);
            } else {
              UninitUse::Branch Branch;
              Branch.Terminator = Term;
              Branch.Output = I - Block->succ_begin();
              Use.addUninitBranch(Branch);
            }
          }
        }
      }
    }

    return Use;
  }
};
}

void TransferFunctions::reportUse(const Expr *ex, const VarDecl *vd) {
  if (!handler)
    return;
  Value v = vals[vd];
  if (isUninitialized(v))
    handler->handleUseOfUninitVariable(vd, getUninitUse(ex, vd, v));
}

void TransferFunctions::VisitObjCForCollectionStmt(ObjCForCollectionStmt *FS) {
  // This represents an initialization of the 'element' value.
  if (DeclStmt *DS = dyn_cast<DeclStmt>(FS->getElement())) {
    const VarDecl *VD = cast<VarDecl>(DS->getSingleDecl());
    if (isTrackedVar(VD))
      vals[VD] = Initialized;
  }
}

void TransferFunctions::VisitBlockExpr(BlockExpr *be) {
  const BlockDecl *bd = be->getBlockDecl();
  for (BlockDecl::capture_const_iterator i = bd->capture_begin(),
        e = bd->capture_end() ; i != e; ++i) {
    const VarDecl *vd = i->getVariable();
    if (!isTrackedVar(vd))
      continue;
    if (i->isByRef()) {
      vals[vd] = Initialized;
      continue;
    }
    reportUse(be, vd);
  }
}

void TransferFunctions::VisitCallExpr(CallExpr *ce) {
  if (Decl *Callee = ce->getCalleeDecl()) {
    if (Callee->hasAttr<ReturnsTwiceAttr>()) {
      // After a call to a function like setjmp or vfork, any variable which is
      // initialized anywhere within this function may now be initialized. For
      // now, just assume such a call initializes all variables.  FIXME: Only
      // mark variables as initialized if they have an initializer which is
      // reachable from here.
      vals.setAllScratchValues(Initialized);
    }
    else if (Callee->hasAttr<AnalyzerNoReturnAttr>()) {
      // Functions labeled like "analyzer_noreturn" are often used to denote
      // "panic" functions that in special debug situations can still return,
      // but for the most part should not be treated as returning.  This is a
      // useful annotation borrowed from the static analyzer that is useful for
      // suppressing branch-specific false positives when we call one of these
      // functions but keep pretending the path continues (when in reality the
      // user doesn't care).
      vals.setAllScratchValues(Unknown);
    }
  }
}

void TransferFunctions::VisitDeclRefExpr(DeclRefExpr *dr) {
  switch (classification.get(dr)) {
  case ClassifyRefs::Ignore:
    break;
  case ClassifyRefs::Use:
    reportUse(dr, cast<VarDecl>(dr->getDecl()));
    break;
  case ClassifyRefs::Init:
    vals[cast<VarDecl>(dr->getDecl())] = Initialized;
    break;
  case ClassifyRefs::SelfInit:
    if (handler)
      handler->handleSelfInit(cast<VarDecl>(dr->getDecl()));
    break;
  }
}

void TransferFunctions::VisitBinaryOperator(BinaryOperator *BO) {
  if (BO->getOpcode() == BO_Assign) {
    FindVarResult Var = findVar(BO->getLHS());
    if (const VarDecl *VD = Var.getDecl())
      vals[VD] = Initialized;
  }
}

void TransferFunctions::VisitDeclStmt(DeclStmt *DS) {
  for (DeclStmt::decl_iterator DI = DS->decl_begin(), DE = DS->decl_end();
       DI != DE; ++DI) {
    VarDecl *VD = dyn_cast<VarDecl>(*DI);
    if (VD && isTrackedVar(VD)) {
      if (getSelfInitExpr(VD)) {
        // If the initializer consists solely of a reference to itself, we
        // explicitly mark the variable as uninitialized. This allows code
        // like the following:
        //
        //   int x = x;
        //
        // to deliberately leave a variable uninitialized. Different analysis
        // clients can detect this pattern and adjust their reporting
        // appropriately, but we need to continue to analyze subsequent uses
        // of the variable.
        vals[VD] = Uninitialized;
      } else if (VD->getInit()) {
        // Treat the new variable as initialized.
        vals[VD] = Initialized;
      } else {
        // No initializer: the variable is now uninitialized. This matters
        // for cases like:
        //   while (...) {
        //     int n;
        //     use(n);
        //     n = 0;
        //   }
        // FIXME: Mark the variable as uninitialized whenever its scope is
        // left, since its scope could be re-entered by a jump over the
        // declaration.
        vals[VD] = Uninitialized;
      }
    }
  }
}

void TransferFunctions::VisitObjCMessageExpr(ObjCMessageExpr *ME) {
  // If the Objective-C message expression is an implicit no-return that
  // is not modeled in the CFG, set the tracked dataflow values to Unknown.
  if (objCNoRet.isImplicitNoReturn(ME)) {
    vals.setAllScratchValues(Unknown);
  }
}

//------------------------------------------------------------------------====//
// High-level "driver" logic for uninitialized values analysis.
//====------------------------------------------------------------------------//

static bool runOnBlock(const CFGBlock *block, const CFG &cfg,
                       AnalysisDeclContext &ac, CFGBlockValues &vals,
                       const ClassifyRefs &classification,
                       llvm::BitVector &wasAnalyzed,
                       UninitVariablesHandler *handler = 0) {
  wasAnalyzed[block->getBlockID()] = true;
  vals.resetScratch();
  // Merge in values of predecessor blocks.
  bool isFirst = true;
  for (CFGBlock::const_pred_iterator I = block->pred_begin(),
       E = block->pred_end(); I != E; ++I) {
    const CFGBlock *pred = *I;
    if (wasAnalyzed[pred->getBlockID()]) {
      vals.mergeIntoScratch(vals.getValueVector(pred), isFirst);
      isFirst = false;
    }
  }
  // Apply the transfer function.
  TransferFunctions tf(vals, cfg, block, ac, classification, handler);
  for (CFGBlock::const_iterator I = block->begin(), E = block->end(); 
       I != E; ++I) {
    if (const CFGStmt *cs = dyn_cast<CFGStmt>(&*I)) {
      tf.Visit(const_cast<Stmt*>(cs->getStmt()));
    }
  }
  return vals.updateValueVectorWithScratch(block);
}

void clang::runUninitializedVariablesAnalysis(
    const DeclContext &dc,
    const CFG &cfg,
    AnalysisDeclContext &ac,
    UninitVariablesHandler &handler,
    UninitVariablesAnalysisStats &stats) {
  CFGBlockValues vals(cfg);
  vals.computeSetOfDeclarations(dc);
  if (vals.hasNoDeclarations())
    return;

  stats.NumVariablesAnalyzed = vals.getNumEntries();

  // Precompute which expressions are uses and which are initializations.
  ClassifyRefs classification(ac);
  cfg.VisitBlockStmts(classification);

  // Mark all variables uninitialized at the entry.
  const CFGBlock &entry = cfg.getEntry();
  ValueVector &vec = vals.getValueVector(&entry);
  const unsigned n = vals.getNumEntries();
  for (unsigned j = 0; j < n ; ++j) {
    vec[j] = Uninitialized;
  }

  // Proceed with the workist.
  DataflowWorklist worklist(cfg);
  llvm::BitVector previouslyVisited(cfg.getNumBlockIDs());
  worklist.enqueueSuccessors(&cfg.getEntry());
  llvm::BitVector wasAnalyzed(cfg.getNumBlockIDs(), false);
  wasAnalyzed[cfg.getEntry().getBlockID()] = true;

  while (const CFGBlock *block = worklist.dequeue()) {
    // Did the block change?
    bool changed = runOnBlock(block, cfg, ac, vals,
                              classification, wasAnalyzed);
    ++stats.NumBlockVisits;
    if (changed || !previouslyVisited[block->getBlockID()])
      worklist.enqueueSuccessors(block);    
    previouslyVisited[block->getBlockID()] = true;
  }
  
  // Run through the blocks one more time, and report uninitialized variabes.
  for (CFG::const_iterator BI = cfg.begin(), BE = cfg.end(); BI != BE; ++BI) {
    const CFGBlock *block = *BI;
    if (wasAnalyzed[block->getBlockID()]) {
      runOnBlock(block, cfg, ac, vals, classification, wasAnalyzed, &handler);
      ++stats.NumBlockVisits;
    }
  }
}

UninitVariablesHandler::~UninitVariablesHandler() {}
