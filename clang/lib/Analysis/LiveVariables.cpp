//=- LiveVariables.cpp - Live Variable Analysis for Source CFGs ----------*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Live Variables analysis for source-level CFGs.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <vector>

using namespace clang;

namespace {

class DataflowWorklist {
  SmallVector<const CFGBlock *, 20> worklist;
  llvm::BitVector enqueuedBlocks;
  PostOrderCFGView *POV;
public:
  DataflowWorklist(const CFG &cfg, AnalysisDeclContext &Ctx)
    : enqueuedBlocks(cfg.getNumBlockIDs()),
      POV(Ctx.getAnalysis<PostOrderCFGView>()) {}
  
  void enqueueBlock(const CFGBlock *block);
  void enqueuePredecessors(const CFGBlock *block);

  const CFGBlock *dequeue();

  void sortWorklist();
};

}

void DataflowWorklist::enqueueBlock(const clang::CFGBlock *block) {
  if (block && !enqueuedBlocks[block->getBlockID()]) {
    enqueuedBlocks[block->getBlockID()] = true;
    worklist.push_back(block);
  }
}

void DataflowWorklist::enqueuePredecessors(const clang::CFGBlock *block) {
  const unsigned OldWorklistSize = worklist.size();
  for (CFGBlock::const_pred_iterator I = block->pred_begin(),
       E = block->pred_end(); I != E; ++I) {
    enqueueBlock(*I);
  }
  
  if (OldWorklistSize == 0 || OldWorklistSize == worklist.size())
    return;

  sortWorklist();
}

void DataflowWorklist::sortWorklist() {
  std::sort(worklist.begin(), worklist.end(), POV->getComparator());
}

const CFGBlock *DataflowWorklist::dequeue() {
  if (worklist.empty())
    return nullptr;
  const CFGBlock *b = worklist.pop_back_val();
  enqueuedBlocks[b->getBlockID()] = false;
  return b;
}

namespace {
class LiveVariablesImpl {
public:  
  AnalysisDeclContext &analysisContext;
  llvm::ImmutableSet<const Stmt *>::Factory SSetFact;
  llvm::ImmutableSet<const VarDecl *>::Factory DSetFact;
  llvm::DenseMap<const CFGBlock *, LiveVariables::LivenessValues> blocksEndToLiveness;
  llvm::DenseMap<const CFGBlock *, LiveVariables::LivenessValues> blocksBeginToLiveness;
  llvm::DenseMap<const Stmt *, LiveVariables::LivenessValues> stmtsToLiveness;
  llvm::DenseMap<const DeclRefExpr *, unsigned> inAssignment;
  const bool killAtAssign;
  
  LiveVariables::LivenessValues
  merge(LiveVariables::LivenessValues valsA,
        LiveVariables::LivenessValues valsB);

  LiveVariables::LivenessValues
  runOnBlock(const CFGBlock *block, LiveVariables::LivenessValues val,
             LiveVariables::Observer *obs = nullptr);

  void dumpBlockLiveness(const SourceManager& M);

  LiveVariablesImpl(AnalysisDeclContext &ac, bool KillAtAssign)
    : analysisContext(ac),
      SSetFact(false), // Do not canonicalize ImmutableSets by default.
      DSetFact(false), // This is a *major* performance win.
      killAtAssign(KillAtAssign) {}
};
}

static LiveVariablesImpl &getImpl(void *x) {
  return *((LiveVariablesImpl *) x);
}

//===----------------------------------------------------------------------===//
// Operations and queries on LivenessValues.
//===----------------------------------------------------------------------===//

bool LiveVariables::LivenessValues::isLive(const Stmt *S) const {
  return liveStmts.contains(S);
}

bool LiveVariables::LivenessValues::isLive(const VarDecl *D) const {
  return liveDecls.contains(D);
}

namespace {
  template <typename SET>
  SET mergeSets(SET A, SET B) {
    if (A.isEmpty())
      return B;
    
    for (typename SET::iterator it = B.begin(), ei = B.end(); it != ei; ++it) {
      A = A.add(*it);
    }
    return A;
  }
}

void LiveVariables::Observer::anchor() { }

LiveVariables::LivenessValues
LiveVariablesImpl::merge(LiveVariables::LivenessValues valsA,
                         LiveVariables::LivenessValues valsB) {  
  
  llvm::ImmutableSetRef<const Stmt *>
    SSetRefA(valsA.liveStmts.getRootWithoutRetain(), SSetFact.getTreeFactory()),
    SSetRefB(valsB.liveStmts.getRootWithoutRetain(), SSetFact.getTreeFactory());
                                                
  
  llvm::ImmutableSetRef<const VarDecl *>
    DSetRefA(valsA.liveDecls.getRootWithoutRetain(), DSetFact.getTreeFactory()),
    DSetRefB(valsB.liveDecls.getRootWithoutRetain(), DSetFact.getTreeFactory());
  

  SSetRefA = mergeSets(SSetRefA, SSetRefB);
  DSetRefA = mergeSets(DSetRefA, DSetRefB);
  
  // asImmutableSet() canonicalizes the tree, allowing us to do an easy
  // comparison afterwards.
  return LiveVariables::LivenessValues(SSetRefA.asImmutableSet(),
                                       DSetRefA.asImmutableSet());  
}

bool LiveVariables::LivenessValues::equals(const LivenessValues &V) const {
  return liveStmts == V.liveStmts && liveDecls == V.liveDecls;
}

//===----------------------------------------------------------------------===//
// Query methods.
//===----------------------------------------------------------------------===//

static bool isAlwaysAlive(const VarDecl *D) {
  return D->hasGlobalStorage();
}

bool LiveVariables::isLive(const CFGBlock *B, const VarDecl *D) {
  return isAlwaysAlive(D) || getImpl(impl).blocksEndToLiveness[B].isLive(D);
}

bool LiveVariables::isLive(const Stmt *S, const VarDecl *D) {
  return isAlwaysAlive(D) || getImpl(impl).stmtsToLiveness[S].isLive(D);
}

bool LiveVariables::isLive(const Stmt *Loc, const Stmt *S) {
  return getImpl(impl).stmtsToLiveness[Loc].isLive(S);
}

//===----------------------------------------------------------------------===//
// Dataflow computation.
//===----------------------------------------------------------------------===//

namespace {
class TransferFunctions : public StmtVisitor<TransferFunctions> {
  LiveVariablesImpl &LV;
  LiveVariables::LivenessValues &val;
  LiveVariables::Observer *observer;
  const CFGBlock *currentBlock;
public:
  TransferFunctions(LiveVariablesImpl &im,
                    LiveVariables::LivenessValues &Val,
                    LiveVariables::Observer *Observer,
                    const CFGBlock *CurrentBlock)
  : LV(im), val(Val), observer(Observer), currentBlock(CurrentBlock) {}

  void VisitBinaryOperator(BinaryOperator *BO);
  void VisitBlockExpr(BlockExpr *BE);
  void VisitDeclRefExpr(DeclRefExpr *DR);  
  void VisitDeclStmt(DeclStmt *DS);
  void VisitObjCForCollectionStmt(ObjCForCollectionStmt *OS);
  void VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *UE);
  void VisitUnaryOperator(UnaryOperator *UO);
  void Visit(Stmt *S);
};
}

static const VariableArrayType *FindVA(QualType Ty) {
  const Type *ty = Ty.getTypePtr();
  while (const ArrayType *VT = dyn_cast<ArrayType>(ty)) {
    if (const VariableArrayType *VAT = dyn_cast<VariableArrayType>(VT))
      if (VAT->getSizeExpr())
        return VAT;
    
    ty = VT->getElementType().getTypePtr();
  }

  return nullptr;
}

static const Stmt *LookThroughStmt(const Stmt *S) {
  while (S) {
    if (const Expr *Ex = dyn_cast<Expr>(S))
      S = Ex->IgnoreParens();    
    if (const ExprWithCleanups *EWC = dyn_cast<ExprWithCleanups>(S)) {
      S = EWC->getSubExpr();
      continue;
    }
    if (const OpaqueValueExpr *OVE = dyn_cast<OpaqueValueExpr>(S)) {
      S = OVE->getSourceExpr();
      continue;
    }
    break;
  }
  return S;
}

static void AddLiveStmt(llvm::ImmutableSet<const Stmt *> &Set,
                        llvm::ImmutableSet<const Stmt *>::Factory &F,
                        const Stmt *S) {
  Set = F.add(Set, LookThroughStmt(S));
}

void TransferFunctions::Visit(Stmt *S) {
  if (observer)
    observer->observeStmt(S, currentBlock, val);
  
  StmtVisitor<TransferFunctions>::Visit(S);
  
  if (isa<Expr>(S)) {
    val.liveStmts = LV.SSetFact.remove(val.liveStmts, S);
  }

  // Mark all children expressions live.
  
  switch (S->getStmtClass()) {
    default:
      break;
    case Stmt::StmtExprClass: {
      // For statement expressions, look through the compound statement.
      S = cast<StmtExpr>(S)->getSubStmt();
      break;
    }
    case Stmt::CXXMemberCallExprClass: {
      // Include the implicit "this" pointer as being live.
      CXXMemberCallExpr *CE = cast<CXXMemberCallExpr>(S);
      if (Expr *ImplicitObj = CE->getImplicitObjectArgument()) {
        AddLiveStmt(val.liveStmts, LV.SSetFact, ImplicitObj);
      }
      break;
    }
    case Stmt::ObjCMessageExprClass: {
      // In calls to super, include the implicit "self" pointer as being live.
      ObjCMessageExpr *CE = cast<ObjCMessageExpr>(S);
      if (CE->getReceiverKind() == ObjCMessageExpr::SuperInstance)
        val.liveDecls = LV.DSetFact.add(val.liveDecls,
                                        LV.analysisContext.getSelfDecl());
      break;
    }
    case Stmt::DeclStmtClass: {
      const DeclStmt *DS = cast<DeclStmt>(S);
      if (const VarDecl *VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
        for (const VariableArrayType* VA = FindVA(VD->getType());
             VA != nullptr; VA = FindVA(VA->getElementType())) {
          AddLiveStmt(val.liveStmts, LV.SSetFact, VA->getSizeExpr());
        }
      }
      break;
    }
    case Stmt::PseudoObjectExprClass: {
      // A pseudo-object operation only directly consumes its result
      // expression.
      Expr *child = cast<PseudoObjectExpr>(S)->getResultExpr();
      if (!child) return;
      if (OpaqueValueExpr *OV = dyn_cast<OpaqueValueExpr>(child))
        child = OV->getSourceExpr();
      child = child->IgnoreParens();
      val.liveStmts = LV.SSetFact.add(val.liveStmts, child);
      return;
    }

    // FIXME: These cases eventually shouldn't be needed.
    case Stmt::ExprWithCleanupsClass: {
      S = cast<ExprWithCleanups>(S)->getSubExpr();
      break;
    }
    case Stmt::CXXBindTemporaryExprClass: {
      S = cast<CXXBindTemporaryExpr>(S)->getSubExpr();
      break;
    }
    case Stmt::UnaryExprOrTypeTraitExprClass: {
      // No need to unconditionally visit subexpressions.
      return;
    }
  }

  for (Stmt *Child : S->children()) {
    if (Child)
      AddLiveStmt(val.liveStmts, LV.SSetFact, Child);
  }
}

void TransferFunctions::VisitBinaryOperator(BinaryOperator *B) {
  if (B->isAssignmentOp()) {
    if (!LV.killAtAssign)
      return;
    
    // Assigning to a variable?
    Expr *LHS = B->getLHS()->IgnoreParens();
    
    if (DeclRefExpr *DR = dyn_cast<DeclRefExpr>(LHS))
      if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
        // Assignments to references don't kill the ref's address
        if (VD->getType()->isReferenceType())
          return;

        if (!isAlwaysAlive(VD)) {
          // The variable is now dead.
          val.liveDecls = LV.DSetFact.remove(val.liveDecls, VD);
        }

        if (observer)
          observer->observerKill(DR);
      }
  }
}

void TransferFunctions::VisitBlockExpr(BlockExpr *BE) {
  for (const VarDecl *VD :
       LV.analysisContext.getReferencedBlockVars(BE->getBlockDecl())) {
    if (isAlwaysAlive(VD))
      continue;
    val.liveDecls = LV.DSetFact.add(val.liveDecls, VD);
  }
}

void TransferFunctions::VisitDeclRefExpr(DeclRefExpr *DR) {
  if (const VarDecl *D = dyn_cast<VarDecl>(DR->getDecl()))
    if (!isAlwaysAlive(D) && LV.inAssignment.find(DR) == LV.inAssignment.end())
      val.liveDecls = LV.DSetFact.add(val.liveDecls, D);
}

void TransferFunctions::VisitDeclStmt(DeclStmt *DS) {
  for (const auto *DI : DS->decls())
    if (const auto *VD = dyn_cast<VarDecl>(DI)) {
      if (!isAlwaysAlive(VD))
        val.liveDecls = LV.DSetFact.remove(val.liveDecls, VD);
    }
}

void TransferFunctions::VisitObjCForCollectionStmt(ObjCForCollectionStmt *OS) {
  // Kill the iteration variable.
  DeclRefExpr *DR = nullptr;
  const VarDecl *VD = nullptr;

  Stmt *element = OS->getElement();
  if (DeclStmt *DS = dyn_cast<DeclStmt>(element)) {
    VD = cast<VarDecl>(DS->getSingleDecl());
  }
  else if ((DR = dyn_cast<DeclRefExpr>(cast<Expr>(element)->IgnoreParens()))) {
    VD = cast<VarDecl>(DR->getDecl());
  }
  
  if (VD) {
    val.liveDecls = LV.DSetFact.remove(val.liveDecls, VD);
    if (observer && DR)
      observer->observerKill(DR);
  }
}

void TransferFunctions::
VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *UE)
{
  // While sizeof(var) doesn't technically extend the liveness of 'var', it
  // does extent the liveness of metadata if 'var' is a VariableArrayType.
  // We handle that special case here.
  if (UE->getKind() != UETT_SizeOf || UE->isArgumentType())
    return;

  const Expr *subEx = UE->getArgumentExpr();
  if (subEx->getType()->isVariableArrayType()) {
    assert(subEx->isLValue());
    val.liveStmts = LV.SSetFact.add(val.liveStmts, subEx->IgnoreParens());
  }
}

void TransferFunctions::VisitUnaryOperator(UnaryOperator *UO) {
  // Treat ++/-- as a kill.
  // Note we don't actually have to do anything if we don't have an observer,
  // since a ++/-- acts as both a kill and a "use".
  if (!observer)
    return;
  
  switch (UO->getOpcode()) {
  default:
    return;
  case UO_PostInc:
  case UO_PostDec:    
  case UO_PreInc:
  case UO_PreDec:
    break;
  }
  
  if (DeclRefExpr *DR = dyn_cast<DeclRefExpr>(UO->getSubExpr()->IgnoreParens()))
    if (isa<VarDecl>(DR->getDecl())) {
      // Treat ++/-- as a kill.
      observer->observerKill(DR);
    }
}

LiveVariables::LivenessValues
LiveVariablesImpl::runOnBlock(const CFGBlock *block,
                              LiveVariables::LivenessValues val,
                              LiveVariables::Observer *obs) {

  TransferFunctions TF(*this, val, obs, block);
  
  // Visit the terminator (if any).
  if (const Stmt *term = block->getTerminator())
    TF.Visit(const_cast<Stmt*>(term));
  
  // Apply the transfer function for all Stmts in the block.
  for (CFGBlock::const_reverse_iterator it = block->rbegin(),
       ei = block->rend(); it != ei; ++it) {
    const CFGElement &elem = *it;

    if (Optional<CFGAutomaticObjDtor> Dtor =
            elem.getAs<CFGAutomaticObjDtor>()) {
      val.liveDecls = DSetFact.add(val.liveDecls, Dtor->getVarDecl());
      continue;
    }

    if (!elem.getAs<CFGStmt>())
      continue;
    
    const Stmt *S = elem.castAs<CFGStmt>().getStmt();
    TF.Visit(const_cast<Stmt*>(S));
    stmtsToLiveness[S] = val;
  }
  return val;
}

void LiveVariables::runOnAllBlocks(LiveVariables::Observer &obs) {
  const CFG *cfg = getImpl(impl).analysisContext.getCFG();
  for (CFG::const_iterator it = cfg->begin(), ei = cfg->end(); it != ei; ++it)
    getImpl(impl).runOnBlock(*it, getImpl(impl).blocksEndToLiveness[*it], &obs);    
}

LiveVariables::LiveVariables(void *im) : impl(im) {} 

LiveVariables::~LiveVariables() {
  delete (LiveVariablesImpl*) impl;
}

LiveVariables *
LiveVariables::computeLiveness(AnalysisDeclContext &AC,
                                 bool killAtAssign) {

  // No CFG?  Bail out.
  CFG *cfg = AC.getCFG();
  if (!cfg)
    return nullptr;

  // The analysis currently has scalability issues for very large CFGs.
  // Bail out if it looks too large.
  if (cfg->getNumBlockIDs() > 300000)
    return nullptr;

  LiveVariablesImpl *LV = new LiveVariablesImpl(AC, killAtAssign);

  // Construct the dataflow worklist.  Enqueue the exit block as the
  // start of the analysis.
  DataflowWorklist worklist(*cfg, AC);
  llvm::BitVector everAnalyzedBlock(cfg->getNumBlockIDs());

  // FIXME: we should enqueue using post order.
  for (CFG::const_iterator it = cfg->begin(), ei = cfg->end(); it != ei; ++it) {
    const CFGBlock *block = *it;
    worklist.enqueueBlock(block);
    
    // FIXME: Scan for DeclRefExprs using in the LHS of an assignment.
    // We need to do this because we lack context in the reverse analysis
    // to determine if a DeclRefExpr appears in such a context, and thus
    // doesn't constitute a "use".
    if (killAtAssign)
      for (CFGBlock::const_iterator bi = block->begin(), be = block->end();
           bi != be; ++bi) {
        if (Optional<CFGStmt> cs = bi->getAs<CFGStmt>()) {
          if (const BinaryOperator *BO =
                  dyn_cast<BinaryOperator>(cs->getStmt())) {
            if (BO->getOpcode() == BO_Assign) {
              if (const DeclRefExpr *DR =
                    dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParens())) {
                LV->inAssignment[DR] = 1;
              }
            }
          }
        }
      }
  }
  
  worklist.sortWorklist();
  
  while (const CFGBlock *block = worklist.dequeue()) {
    // Determine if the block's end value has changed.  If not, we
    // have nothing left to do for this block.
    LivenessValues &prevVal = LV->blocksEndToLiveness[block];
    
    // Merge the values of all successor blocks.
    LivenessValues val;
    for (CFGBlock::const_succ_iterator it = block->succ_begin(),
                                       ei = block->succ_end(); it != ei; ++it) {
      if (const CFGBlock *succ = *it) {     
        val = LV->merge(val, LV->blocksBeginToLiveness[succ]);
      }
    }
    
    if (!everAnalyzedBlock[block->getBlockID()])
      everAnalyzedBlock[block->getBlockID()] = true;
    else if (prevVal.equals(val))
      continue;

    prevVal = val;
    
    // Update the dataflow value for the start of this block.
    LV->blocksBeginToLiveness[block] = LV->runOnBlock(block, val);
    
    // Enqueue the value to the predecessors.
    worklist.enqueuePredecessors(block);
  }
  
  return new LiveVariables(LV);
}

void LiveVariables::dumpBlockLiveness(const SourceManager &M) {
  getImpl(impl).dumpBlockLiveness(M);
}

void LiveVariablesImpl::dumpBlockLiveness(const SourceManager &M) {
  std::vector<const CFGBlock *> vec;
  for (llvm::DenseMap<const CFGBlock *, LiveVariables::LivenessValues>::iterator
       it = blocksEndToLiveness.begin(), ei = blocksEndToLiveness.end();
       it != ei; ++it) {
    vec.push_back(it->first);    
  }
  std::sort(vec.begin(), vec.end(), [](const CFGBlock *A, const CFGBlock *B) {
    return A->getBlockID() < B->getBlockID();
  });

  std::vector<const VarDecl*> declVec;

  for (std::vector<const CFGBlock *>::iterator
        it = vec.begin(), ei = vec.end(); it != ei; ++it) {
    llvm::errs() << "\n[ B" << (*it)->getBlockID()
                 << " (live variables at block exit) ]\n";
    
    LiveVariables::LivenessValues vals = blocksEndToLiveness[*it];
    declVec.clear();
    
    for (llvm::ImmutableSet<const VarDecl *>::iterator si =
          vals.liveDecls.begin(),
          se = vals.liveDecls.end(); si != se; ++si) {
      declVec.push_back(*si);      
    }

    std::sort(declVec.begin(), declVec.end(), [](const Decl *A, const Decl *B) {
      return A->getLocStart() < B->getLocStart();
    });

    for (std::vector<const VarDecl*>::iterator di = declVec.begin(),
         de = declVec.end(); di != de; ++di) {
      llvm::errs() << " " << (*di)->getDeclName().getAsString()
                   << " <";
      (*di)->getLocation().dump(M);
      llvm::errs() << ">\n";
    }
  }
  llvm::errs() << "\n";  
}

const void *LiveVariables::getTag() { static int x; return &x; }
const void *RelaxedLiveVariables::getTag() { static int x; return &x; }
