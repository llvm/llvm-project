//===- Consumed.h ----------------------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A intra-procedural analysis for checking consumed properties.  This is based,
// in part, on research on linear types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_CONSUMED_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_CONSUMED_H

#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {
namespace consumed {
  
  enum ConsumedState {
    // No state information for the given variable.
    CS_None,
    
    CS_Unknown,
    CS_Unconsumed,
    CS_Consumed
  };
  
  class ConsumedStmtVisitor;
  
  typedef SmallVector<PartialDiagnosticAt, 1> OptionalNotes;
  typedef std::pair<PartialDiagnosticAt, OptionalNotes> DelayedDiag;
  typedef std::list<DelayedDiag> DiagList;

  class ConsumedWarningsHandlerBase {

  public:

    virtual ~ConsumedWarningsHandlerBase();

    /// \brief Emit the warnings and notes left by the analysis.
    virtual void emitDiagnostics() {}
    
    /// \brief Warn that a variable's state doesn't match at the entry and exit
    /// of a loop.
    ///
    /// \param Loc -- The location of the end of the loop.
    ///
    /// \param VariableName -- The name of the variable that has a mismatched
    /// state.
    virtual void warnLoopStateMismatch(SourceLocation Loc,
                                       StringRef VariableName) {}
    
    /// \brief Warn about parameter typestate mismatches upon return.
    ///
    /// \param Loc -- The SourceLocation of the return statement.
    ///
    /// \param ExpectedState -- The state the return value was expected to be
    /// in.
    ///
    /// \param ObservedState -- The state the return value was observed to be
    /// in.
    virtual void warnParamReturnTypestateMismatch(SourceLocation Loc,
                                                  StringRef VariableName,
                                                  StringRef ExpectedState,
                                                  StringRef ObservedState) {}
    
    // FIXME: Add documentation.
    virtual void warnParamTypestateMismatch(SourceLocation LOC,
                                            StringRef ExpectedState,
                                            StringRef ObservedState) {}
    
    // FIXME: This can be removed when the attr propagation fix for templated
    //        classes lands.
    /// \brief Warn about return typestates set for unconsumable types.
    /// 
    /// \param Loc -- The location of the attributes.
    ///
    /// \param TypeName -- The name of the unconsumable type.
    virtual void warnReturnTypestateForUnconsumableType(SourceLocation Loc,
                                                        StringRef TypeName) {}
    
    /// \brief Warn about return typestate mismatches.
    ///
    /// \param Loc -- The SourceLocation of the return statement.
    ///
    /// \param ExpectedState -- The state the return value was expected to be
    /// in.
    ///
    /// \param ObservedState -- The state the return value was observed to be
    /// in.
    virtual void warnReturnTypestateMismatch(SourceLocation Loc,
                                             StringRef ExpectedState,
                                             StringRef ObservedState) {}

    /// \brief Warn about use-while-consumed errors.
    /// \param MethodName -- The name of the method that was incorrectly
    /// invoked.
    ///
    /// \param State -- The state the object was used in.
    ///
    /// \param Loc -- The SourceLocation of the method invocation.
    virtual void warnUseOfTempInInvalidState(StringRef MethodName,
                                             StringRef State,
                                             SourceLocation Loc) {}

    /// \brief Warn about use-while-consumed errors.
    /// \param MethodName -- The name of the method that was incorrectly
    /// invoked.
    ///
    /// \param State -- The state the object was used in.
    ///
    /// \param VariableName -- The name of the variable that holds the unique
    /// value.
    ///
    /// \param Loc -- The SourceLocation of the method invocation.
    virtual void warnUseInInvalidState(StringRef MethodName,
                                       StringRef VariableName,
                                       StringRef State,
                                       SourceLocation Loc) {}
  };

  class ConsumedStateMap {
    
    typedef llvm::DenseMap<const VarDecl *, ConsumedState> VarMapType;
    typedef llvm::DenseMap<const CXXBindTemporaryExpr *, ConsumedState>
            TmpMapType;
    
  protected:
    
    bool Reachable;
    const Stmt *From;
    VarMapType VarMap;
    TmpMapType TmpMap;
    
  public:
    ConsumedStateMap() : Reachable(true), From(nullptr) {}
    ConsumedStateMap(const ConsumedStateMap &Other)
      : Reachable(Other.Reachable), From(Other.From), VarMap(Other.VarMap),
        TmpMap() {}
    
    /// \brief Warn if any of the parameters being tracked are not in the state
    /// they were declared to be in upon return from a function.
    void checkParamsForReturnTypestate(SourceLocation BlameLoc,
      ConsumedWarningsHandlerBase &WarningsHandler) const;
    
    /// \brief Clear the TmpMap.
    void clearTemporaries();
    
    /// \brief Get the consumed state of a given variable.
    ConsumedState getState(const VarDecl *Var) const;
    
    /// \brief Get the consumed state of a given temporary value.
    ConsumedState getState(const CXXBindTemporaryExpr *Tmp) const;
    
    /// \brief Merge this state map with another map.
    void intersect(const ConsumedStateMap &Other);

    void intersectAtLoopHead(const CFGBlock *LoopHead, const CFGBlock *LoopBack,
      const ConsumedStateMap *LoopBackStates,
      ConsumedWarningsHandlerBase &WarningsHandler);
    
    /// \brief Return true if this block is reachable.
    bool isReachable() const { return Reachable; }
    
    /// \brief Mark the block as unreachable.
    void markUnreachable();
    
    /// \brief Set the source for a decision about the branching of states.
    /// \param Source -- The statement that was the origin of a branching
    /// decision.
    void setSource(const Stmt *Source) { this->From = Source; }
    
    /// \brief Set the consumed state of a given variable.
    void setState(const VarDecl *Var, ConsumedState State);
    
    /// \brief Set the consumed state of a given temporary value.
    void setState(const CXXBindTemporaryExpr *Tmp, ConsumedState State);
    
    /// \brief Remove the temporary value from our state map.
    void remove(const CXXBindTemporaryExpr *Tmp);
    
    /// \brief Tests to see if there is a mismatch in the states stored in two
    /// maps.
    ///
    /// \param Other -- The second map to compare against.
    bool operator!=(const ConsumedStateMap *Other) const;
  };
  
  class ConsumedBlockInfo {
    std::vector<std::unique_ptr<ConsumedStateMap>> StateMapsArray;
    std::vector<unsigned int> VisitOrder;
    
  public:
    ConsumedBlockInfo() = default;

    ConsumedBlockInfo(unsigned int NumBlocks, PostOrderCFGView *SortedGraph)
        : StateMapsArray(NumBlocks), VisitOrder(NumBlocks, 0) {
      unsigned int VisitOrderCounter = 0;
      for (PostOrderCFGView::iterator BI = SortedGraph->begin(),
           BE = SortedGraph->end(); BI != BE; ++BI) {
        VisitOrder[(*BI)->getBlockID()] = VisitOrderCounter++;
      }
    }
    
    bool allBackEdgesVisited(const CFGBlock *CurrBlock,
                             const CFGBlock *TargetBlock);

    void addInfo(const CFGBlock *Block, ConsumedStateMap *StateMap,
                 std::unique_ptr<ConsumedStateMap> &OwnedStateMap);
    void addInfo(const CFGBlock *Block,
                 std::unique_ptr<ConsumedStateMap> StateMap);

    ConsumedStateMap* borrowInfo(const CFGBlock *Block);
    
    void discardInfo(const CFGBlock *Block);

    std::unique_ptr<ConsumedStateMap> getInfo(const CFGBlock *Block);

    bool isBackEdge(const CFGBlock *From, const CFGBlock *To);
    bool isBackEdgeTarget(const CFGBlock *Block);
  };

  /// A class that handles the analysis of uniqueness violations.
  class ConsumedAnalyzer {
    
    ConsumedBlockInfo BlockInfo;
    std::unique_ptr<ConsumedStateMap> CurrStates;

    ConsumedState ExpectedReturnState;
    
    void determineExpectedReturnState(AnalysisDeclContext &AC,
                                      const FunctionDecl *D);
    bool splitState(const CFGBlock *CurrBlock,
                    const ConsumedStmtVisitor &Visitor);
    
  public:
    
    ConsumedWarningsHandlerBase &WarningsHandler;

    ConsumedAnalyzer(ConsumedWarningsHandlerBase &WarningsHandler)
        : WarningsHandler(WarningsHandler) {}

    ConsumedState getExpectedReturnState() const { return ExpectedReturnState; }
    
    /// \brief Check a function's CFG for consumed violations.
    ///
    /// We traverse the blocks in the CFG, keeping track of the state of each
    /// value who's type has uniquness annotations.  If methods are invoked in
    /// the wrong state a warning is issued.  Each block in the CFG is traversed
    /// exactly once.
    void run(AnalysisDeclContext &AC);
  };
}} // end namespace clang::consumed

#endif
