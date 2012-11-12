//===-- ExprEngine.h - Path-Sensitive Expression-Level Dataflow ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a meta-engine for path-sensitive dataflow analysis that
//  is built on CoreEngine, but provides the boilerplate to execute transfer
//  functions and build the ExplodedGraph at the expression level.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_EXPRENGINE
#define LLVM_CLANG_GR_EXPRENGINE

#include "clang/Analysis/DomainSpecific/ObjCNoReturn.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SubEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CoreEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"

namespace clang {

class AnalysisDeclContextManager;
class CXXCatchStmt;
class CXXConstructExpr;
class CXXDeleteExpr;
class CXXNewExpr;
class CXXTemporaryObjectExpr;
class CXXThisExpr;
class MaterializeTemporaryExpr;
class ObjCAtSynchronizedStmt;
class ObjCForCollectionStmt;
  
namespace ento {

class AnalysisManager;
class CallEvent;
class SimpleCall;

class ExprEngine : public SubEngine {
  AnalysisManager &AMgr;
  
  AnalysisDeclContextManager &AnalysisDeclContexts;

  CoreEngine Engine;

  /// G - the simulation graph.
  ExplodedGraph& G;

  /// StateMgr - Object that manages the data for all created states.
  ProgramStateManager StateMgr;

  /// SymMgr - Object that manages the symbol information.
  SymbolManager& SymMgr;

  /// svalBuilder - SValBuilder object that creates SVals from expressions.
  SValBuilder &svalBuilder;

  /// EntryNode - The immediate predecessor node.
  ExplodedNode *EntryNode;

  /// CleanedState - The state for EntryNode "cleaned" of all dead
  ///  variables and symbols (as determined by a liveness analysis).
  ProgramStateRef CleanedState;

  /// currStmt - The current block-level statement.
  const Stmt *currStmt;
  unsigned int currStmtIdx;
  const NodeBuilderContext *currBldrCtx;
  
  /// Helper object to determine if an Objective-C message expression
  /// implicitly never returns.
  ObjCNoReturn ObjCNoRet;
  
  /// Whether or not GC is enabled in this analysis.
  bool ObjCGCEnabled;

  /// The BugReporter associated with this engine.  It is important that
  ///  this object be placed at the very end of member variables so that its
  ///  destructor is called before the rest of the ExprEngine is destroyed.
  GRBugReporter BR;

  /// The functions which have been analyzed through inlining. This is owned by
  /// AnalysisConsumer. It can be null.
  SetOfConstDecls *VisitedCallees;

public:
  ExprEngine(AnalysisManager &mgr, bool gcEnabled,
             SetOfConstDecls *VisitedCalleesIn,
             FunctionSummariesTy *FS);

  ~ExprEngine();

  /// Returns true if there is still simulation state on the worklist.
  bool ExecuteWorkList(const LocationContext *L, unsigned Steps = 150000) {
    return Engine.ExecuteWorkList(L, Steps, 0);
  }

  /// Execute the work list with an initial state. Nodes that reaches the exit
  /// of the function are added into the Dst set, which represent the exit
  /// state of the function call. Returns true if there is still simulation
  /// state on the worklist.
  bool ExecuteWorkListWithInitialState(const LocationContext *L, unsigned Steps,
                                       ProgramStateRef InitState, 
                                       ExplodedNodeSet &Dst) {
    return Engine.ExecuteWorkListWithInitialState(L, Steps, InitState, Dst);
  }

  /// getContext - Return the ASTContext associated with this analysis.
  ASTContext &getContext() const { return AMgr.getASTContext(); }

  virtual AnalysisManager &getAnalysisManager() { return AMgr; }

  CheckerManager &getCheckerManager() const {
    return *AMgr.getCheckerManager();
  }

  SValBuilder &getSValBuilder() { return svalBuilder; }

  BugReporter& getBugReporter() { return BR; }

  const NodeBuilderContext &getBuilderContext() {
    assert(currBldrCtx);
    return *currBldrCtx;
  }

  bool isObjCGCEnabled() { return ObjCGCEnabled; }

  const Stmt *getStmt() const;

  void GenerateAutoTransition(ExplodedNode *N);
  void enqueueEndOfPath(ExplodedNodeSet &S);
  void GenerateCallExitNode(ExplodedNode *N);

  /// ViewGraph - Visualize the ExplodedGraph created by executing the
  ///  simulation.
  void ViewGraph(bool trim = false);

  void ViewGraph(ExplodedNode** Beg, ExplodedNode** End);

  /// getInitialState - Return the initial state used for the root vertex
  ///  in the ExplodedGraph.
  ProgramStateRef getInitialState(const LocationContext *InitLoc);

  ExplodedGraph& getGraph() { return G; }
  const ExplodedGraph& getGraph() const { return G; }

  /// \brief Run the analyzer's garbage collection - remove dead symbols and
  /// bindings.
  ///
  /// \param Node - The predecessor node, from which the processing should 
  /// start.
  /// \param Out - The returned set of output nodes.
  /// \param ReferenceStmt - Run garbage collection using the symbols, 
  /// which are live before the given statement.
  /// \param LC - The location context of the ReferenceStmt.
  /// \param DiagnosticStmt - the statement used to associate the diagnostic 
  /// message, if any warnings should occur while removing the dead (leaks 
  /// are usually reported here).
  /// \param K - In some cases it is possible to use PreStmt kind. (Do 
  /// not use it unless you know what you are doing.) 
  /// If the ReferenceStmt is NULL, everything is this and parent contexts is
  /// considered live.
  /// If the stack frame context is NULL, everything on stack is considered
  /// dead.
  void removeDead(ExplodedNode *Node, ExplodedNodeSet &Out,
            const Stmt *ReferenceStmt, const StackFrameContext *LC,
            const Stmt *DiagnosticStmt,
            ProgramPoint::Kind K = ProgramPoint::PreStmtPurgeDeadSymbolsKind);

  /// processCFGElement - Called by CoreEngine. Used to generate new successor
  ///  nodes by processing the 'effects' of a CFG element.
  void processCFGElement(const CFGElement E, ExplodedNode *Pred,
                         unsigned StmtIdx, NodeBuilderContext *Ctx);

  void ProcessStmt(const CFGStmt S, ExplodedNode *Pred);

  void ProcessInitializer(const CFGInitializer I, ExplodedNode *Pred);

  void ProcessImplicitDtor(const CFGImplicitDtor D, ExplodedNode *Pred);

  void ProcessAutomaticObjDtor(const CFGAutomaticObjDtor D, 
                               ExplodedNode *Pred, ExplodedNodeSet &Dst);
  void ProcessBaseDtor(const CFGBaseDtor D,
                       ExplodedNode *Pred, ExplodedNodeSet &Dst);
  void ProcessMemberDtor(const CFGMemberDtor D,
                         ExplodedNode *Pred, ExplodedNodeSet &Dst);
  void ProcessTemporaryDtor(const CFGTemporaryDtor D, 
                            ExplodedNode *Pred, ExplodedNodeSet &Dst);

  /// Called by CoreEngine when processing the entrance of a CFGBlock.
  virtual void processCFGBlockEntrance(const BlockEdge &L,
                                       NodeBuilderWithSinks &nodeBuilder,
                                       ExplodedNode *Pred);
  
  /// ProcessBranch - Called by CoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a branch condition.
  void processBranch(const Stmt *Condition, const Stmt *Term, 
                     NodeBuilderContext& BuilderCtx,
                     ExplodedNode *Pred,
                     ExplodedNodeSet &Dst,
                     const CFGBlock *DstT,
                     const CFGBlock *DstF);

  /// processIndirectGoto - Called by CoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a computed goto jump.
  void processIndirectGoto(IndirectGotoNodeBuilder& builder);

  /// ProcessSwitch - Called by CoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a switch statement.
  void processSwitch(SwitchNodeBuilder& builder);

  /// ProcessEndPath - Called by CoreEngine.  Used to generate end-of-path
  ///  nodes when the control reaches the end of a function.
  void processEndOfFunction(NodeBuilderContext& BC,
                            ExplodedNode *Pred);

  /// Remove dead bindings/symbols before exiting a function.
  void removeDeadOnEndOfFunction(NodeBuilderContext& BC,
                                 ExplodedNode *Pred,
                                 ExplodedNodeSet &Dst);

  /// Generate the entry node of the callee.
  void processCallEnter(CallEnter CE, ExplodedNode *Pred);

  /// Generate the sequence of nodes that simulate the call exit and the post
  /// visit for CallExpr.
  void processCallExit(ExplodedNode *Pred);

  /// Called by CoreEngine when the analysis worklist has terminated.
  void processEndWorklist(bool hasWorkRemaining);

  /// evalAssume - Callback function invoked by the ConstraintManager when
  ///  making assumptions about state values.
  ProgramStateRef processAssume(ProgramStateRef state, SVal cond,bool assumption);

  /// wantsRegionChangeUpdate - Called by ProgramStateManager to determine if a
  ///  region change should trigger a processRegionChanges update.
  bool wantsRegionChangeUpdate(ProgramStateRef state);

  /// processRegionChanges - Called by ProgramStateManager whenever a change is made
  ///  to the store. Used to update checkers that track region values.
  ProgramStateRef 
  processRegionChanges(ProgramStateRef state,
                       const StoreManager::InvalidatedSymbols *invalidated,
                       ArrayRef<const MemRegion *> ExplicitRegions,
                       ArrayRef<const MemRegion *> Regions,
                       const CallEvent *Call);

  /// printState - Called by ProgramStateManager to print checker-specific data.
  void printState(raw_ostream &Out, ProgramStateRef State,
                  const char *NL, const char *Sep);

  virtual ProgramStateManager& getStateManager() { return StateMgr; }

  StoreManager& getStoreManager() { return StateMgr.getStoreManager(); }

  ConstraintManager& getConstraintManager() {
    return StateMgr.getConstraintManager();
  }

  // FIXME: Remove when we migrate over to just using SValBuilder.
  BasicValueFactory& getBasicVals() {
    return StateMgr.getBasicVals();
  }

  // FIXME: Remove when we migrate over to just using ValueManager.
  SymbolManager& getSymbolManager() { return SymMgr; }
  const SymbolManager& getSymbolManager() const { return SymMgr; }

  // Functions for external checking of whether we have unfinished work
  bool wasBlocksExhausted() const { return Engine.wasBlocksExhausted(); }
  bool hasEmptyWorkList() const { return !Engine.getWorkList()->hasWork(); }
  bool hasWorkRemaining() const { return Engine.hasWorkRemaining(); }

  const CoreEngine &getCoreEngine() const { return Engine; }

public:
  /// Visit - Transfer function logic for all statements.  Dispatches to
  ///  other functions that handle specific kinds of statements.
  void Visit(const Stmt *S, ExplodedNode *Pred, ExplodedNodeSet &Dst);

  /// VisitArraySubscriptExpr - Transfer function for array accesses.
  void VisitLvalArraySubscriptExpr(const ArraySubscriptExpr *Ex,
                                   ExplodedNode *Pred,
                                   ExplodedNodeSet &Dst);

  /// VisitGCCAsmStmt - Transfer function logic for inline asm.
  void VisitGCCAsmStmt(const GCCAsmStmt *A, ExplodedNode *Pred,
                       ExplodedNodeSet &Dst);

  /// VisitMSAsmStmt - Transfer function logic for MS inline asm.
  void VisitMSAsmStmt(const MSAsmStmt *A, ExplodedNode *Pred,
                      ExplodedNodeSet &Dst);

  /// VisitBlockExpr - Transfer function logic for BlockExprs.
  void VisitBlockExpr(const BlockExpr *BE, ExplodedNode *Pred, 
                      ExplodedNodeSet &Dst);

  /// VisitBinaryOperator - Transfer function logic for binary operators.
  void VisitBinaryOperator(const BinaryOperator* B, ExplodedNode *Pred, 
                           ExplodedNodeSet &Dst);


  /// VisitCall - Transfer function for function calls.
  void VisitCallExpr(const CallExpr *CE, ExplodedNode *Pred,
                     ExplodedNodeSet &Dst);

  /// VisitCast - Transfer function logic for all casts (implicit and explicit).
  void VisitCast(const CastExpr *CastE, const Expr *Ex, ExplodedNode *Pred,
                ExplodedNodeSet &Dst);

  /// VisitCompoundLiteralExpr - Transfer function logic for compound literals.
  void VisitCompoundLiteralExpr(const CompoundLiteralExpr *CL, 
                                ExplodedNode *Pred, ExplodedNodeSet &Dst);

  /// Transfer function logic for DeclRefExprs and BlockDeclRefExprs.
  void VisitCommonDeclRefExpr(const Expr *DR, const NamedDecl *D,
                              ExplodedNode *Pred, ExplodedNodeSet &Dst);
  
  /// VisitDeclStmt - Transfer function logic for DeclStmts.
  void VisitDeclStmt(const DeclStmt *DS, ExplodedNode *Pred, 
                     ExplodedNodeSet &Dst);

  /// VisitGuardedExpr - Transfer function logic for ?, __builtin_choose
  void VisitGuardedExpr(const Expr *Ex, const Expr *L, const Expr *R, 
                        ExplodedNode *Pred, ExplodedNodeSet &Dst);
  
  void VisitInitListExpr(const InitListExpr *E, ExplodedNode *Pred,
                         ExplodedNodeSet &Dst);

  /// VisitLogicalExpr - Transfer function logic for '&&', '||'
  void VisitLogicalExpr(const BinaryOperator* B, ExplodedNode *Pred,
                        ExplodedNodeSet &Dst);

  /// VisitMemberExpr - Transfer function for member expressions.
  void VisitMemberExpr(const MemberExpr *M, ExplodedNode *Pred, 
                           ExplodedNodeSet &Dst);

  /// Transfer function logic for ObjCAtSynchronizedStmts.
  void VisitObjCAtSynchronizedStmt(const ObjCAtSynchronizedStmt *S,
                                   ExplodedNode *Pred, ExplodedNodeSet &Dst);

  /// Transfer function logic for computing the lvalue of an Objective-C ivar.
  void VisitLvalObjCIvarRefExpr(const ObjCIvarRefExpr *DR, ExplodedNode *Pred,
                                ExplodedNodeSet &Dst);

  /// VisitObjCForCollectionStmt - Transfer function logic for
  ///  ObjCForCollectionStmt.
  void VisitObjCForCollectionStmt(const ObjCForCollectionStmt *S, 
                                  ExplodedNode *Pred, ExplodedNodeSet &Dst);

  void VisitObjCMessage(const ObjCMessageExpr *ME, ExplodedNode *Pred,
                        ExplodedNodeSet &Dst);

  /// VisitReturnStmt - Transfer function logic for return statements.
  void VisitReturnStmt(const ReturnStmt *R, ExplodedNode *Pred, 
                       ExplodedNodeSet &Dst);
  
  /// VisitOffsetOfExpr - Transfer function for offsetof.
  void VisitOffsetOfExpr(const OffsetOfExpr *Ex, ExplodedNode *Pred,
                         ExplodedNodeSet &Dst);

  /// VisitUnaryExprOrTypeTraitExpr - Transfer function for sizeof.
  void VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *Ex,
                              ExplodedNode *Pred, ExplodedNodeSet &Dst);

  /// VisitUnaryOperator - Transfer function logic for unary operators.
  void VisitUnaryOperator(const UnaryOperator* B, ExplodedNode *Pred, 
                          ExplodedNodeSet &Dst);

  /// Handle ++ and -- (both pre- and post-increment).
  void VisitIncrementDecrementOperator(const UnaryOperator* U,
                                       ExplodedNode *Pred,
                                       ExplodedNodeSet &Dst);
  
  void VisitCXXCatchStmt(const CXXCatchStmt *CS, ExplodedNode *Pred,
                         ExplodedNodeSet &Dst);

  void VisitCXXThisExpr(const CXXThisExpr *TE, ExplodedNode *Pred, 
                        ExplodedNodeSet & Dst);

  void VisitCXXConstructExpr(const CXXConstructExpr *E, ExplodedNode *Pred,
                             ExplodedNodeSet &Dst);

  void VisitCXXDestructor(QualType ObjectType, const MemRegion *Dest,
                          const Stmt *S, bool IsBaseDtor,
                          ExplodedNode *Pred, ExplodedNodeSet &Dst);

  void VisitCXXNewExpr(const CXXNewExpr *CNE, ExplodedNode *Pred,
                       ExplodedNodeSet &Dst);

  void VisitCXXDeleteExpr(const CXXDeleteExpr *CDE, ExplodedNode *Pred,
                          ExplodedNodeSet &Dst);

  /// Create a C++ temporary object for an rvalue.
  void CreateCXXTemporaryObject(const MaterializeTemporaryExpr *ME,
                                ExplodedNode *Pred, 
                                ExplodedNodeSet &Dst);
  
  /// evalEagerlyAssumeBinOpBifurcation - Given the nodes in 'Src', eagerly assume symbolic
  ///  expressions of the form 'x != 0' and generate new nodes (stored in Dst)
  ///  with those assumptions.
  void evalEagerlyAssumeBinOpBifurcation(ExplodedNodeSet &Dst, ExplodedNodeSet &Src, 
                         const Expr *Ex);
  
  std::pair<const ProgramPointTag *, const ProgramPointTag*>
    geteagerlyAssumeBinOpBifurcationTags();

  SVal evalMinus(SVal X) {
    return X.isValid() ? svalBuilder.evalMinus(cast<NonLoc>(X)) : X;
  }

  SVal evalComplement(SVal X) {
    return X.isValid() ? svalBuilder.evalComplement(cast<NonLoc>(X)) : X;
  }

public:

  SVal evalBinOp(ProgramStateRef state, BinaryOperator::Opcode op,
                 NonLoc L, NonLoc R, QualType T) {
    return svalBuilder.evalBinOpNN(state, op, L, R, T);
  }

  SVal evalBinOp(ProgramStateRef state, BinaryOperator::Opcode op,
                 NonLoc L, SVal R, QualType T) {
    return R.isValid() ? svalBuilder.evalBinOpNN(state,op,L, cast<NonLoc>(R), T) : R;
  }

  SVal evalBinOp(ProgramStateRef ST, BinaryOperator::Opcode Op,
                 SVal LHS, SVal RHS, QualType T) {
    return svalBuilder.evalBinOp(ST, Op, LHS, RHS, T);
  }
  
protected:
  /// evalBind - Handle the semantics of binding a value to a specific location.
  ///  This method is used by evalStore, VisitDeclStmt, and others.
  void evalBind(ExplodedNodeSet &Dst, const Stmt *StoreE, ExplodedNode *Pred,
                SVal location, SVal Val, bool atDeclInit = false,
                const ProgramPoint *PP = 0);

public:
  // FIXME: 'tag' should be removed, and a LocationContext should be used
  // instead.
  // FIXME: Comment on the meaning of the arguments, when 'St' may not
  // be the same as Pred->state, and when 'location' may not be the
  // same as state->getLValue(Ex).
  /// Simulate a read of the result of Ex.
  void evalLoad(ExplodedNodeSet &Dst,
                const Expr *NodeEx,  /* Eventually will be a CFGStmt */
                const Expr *BoundExpr,
                ExplodedNode *Pred,
                ProgramStateRef St,
                SVal location,
                const ProgramPointTag *tag = 0,
                QualType LoadTy = QualType());

  // FIXME: 'tag' should be removed, and a LocationContext should be used
  // instead.
  void evalStore(ExplodedNodeSet &Dst, const Expr *AssignE, const Expr *StoreE,
                 ExplodedNode *Pred, ProgramStateRef St, SVal TargetLV, SVal Val,
                 const ProgramPointTag *tag = 0);

  /// \brief Create a new state in which the call return value is binded to the
  /// call origin expression.
  ProgramStateRef bindReturnValue(const CallEvent &Call,
                                  const LocationContext *LCtx,
                                  ProgramStateRef State);

  /// Evaluate a call, running pre- and post-call checks and allowing checkers
  /// to be responsible for handling the evaluation of the call itself.
  void evalCall(ExplodedNodeSet &Dst, ExplodedNode *Pred,
                const CallEvent &Call);

  /// \brief Default implementation of call evaluation.
  void defaultEvalCall(NodeBuilder &B, ExplodedNode *Pred,
                       const CallEvent &Call);
private:
  void evalLoadCommon(ExplodedNodeSet &Dst,
                      const Expr *NodeEx,  /* Eventually will be a CFGStmt */
                      const Expr *BoundEx,
                      ExplodedNode *Pred,
                      ProgramStateRef St,
                      SVal location,
                      const ProgramPointTag *tag,
                      QualType LoadTy);

  // FIXME: 'tag' should be removed, and a LocationContext should be used
  // instead.
  void evalLocation(ExplodedNodeSet &Dst,
                    const Stmt *NodeEx, /* This will eventually be a CFGStmt */
                    const Stmt *BoundEx,
                    ExplodedNode *Pred,
                    ProgramStateRef St, SVal location,
                    const ProgramPointTag *tag, bool isLoad);

  /// Count the stack depth and determine if the call is recursive.
  void examineStackFrames(const Decl *D, const LocationContext *LCtx,
                          bool &IsRecursive, unsigned &StackDepth);

  bool shouldInlineDecl(const Decl *D, ExplodedNode *Pred);
  bool inlineCall(const CallEvent &Call, const Decl *D, NodeBuilder &Bldr,
                  ExplodedNode *Pred, ProgramStateRef State);

  /// \brief Conservatively evaluate call by invalidating regions and binding
  /// a conjured return value.
  void conservativeEvalCall(const CallEvent &Call, NodeBuilder &Bldr,
                            ExplodedNode *Pred, ProgramStateRef State);

  /// \brief Either inline or process the call conservatively (or both), based
  /// on DynamicDispatchBifurcation data.
  void BifurcateCall(const MemRegion *BifurReg,
                     const CallEvent &Call, const Decl *D, NodeBuilder &Bldr,
                     ExplodedNode *Pred);

  bool replayWithoutInlining(ExplodedNode *P, const LocationContext *CalleeLC);
};

/// Traits for storing the call processing policy inside GDM.
/// The GDM stores the corresponding CallExpr pointer.
// FIXME: This does not use the nice trait macros because it must be accessible
// from multiple translation units.
struct ReplayWithoutInlining{};
template <>
struct ProgramStateTrait<ReplayWithoutInlining> :
  public ProgramStatePartialTrait<void*> {
  static void *GDMIndex() { static int index = 0; return &index; }
};

} // end ento namespace

} // end clang namespace

#endif
