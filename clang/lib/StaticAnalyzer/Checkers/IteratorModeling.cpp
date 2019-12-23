//===-- IteratorModeling.cpp --------------------------------------*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a checker for using iterators outside their range (past end). Usage
// means here dereferencing, incrementing etc.
//
//===----------------------------------------------------------------------===//
//
// In the code, iterator can be represented as a:
// * type-I: typedef-ed pointer. Operations over such iterator, such as
//           comparisons or increments, are modeled straightforwardly by the
//           analyzer.
// * type-II: structure with its method bodies available.  Operations over such
//            iterator are inlined by the analyzer, and results of modeling
//            these operations are exposing implementation details of the
//            iterators, which is not necessarily helping.
// * type-III: completely opaque structure. Operations over such iterator are
//             modeled conservatively, producing conjured symbols everywhere.
//
// To handle all these types in a common way we introduce a structure called
// IteratorPosition which is an abstraction of the position the iterator
// represents using symbolic expressions. The checker handles all the
// operations on this structure.
//
// Additionally, depending on the circumstances, operators of types II and III
// can be represented as:
// * type-IIa, type-IIIa: conjured structure symbols - when returned by value
//                        from conservatively evaluated methods such as
//                        `.begin()`.
// * type-IIb, type-IIIb: memory regions of iterator-typed objects, such as
//                        variables or temporaries, when the iterator object is
//                        currently treated as an lvalue.
// * type-IIc, type-IIIc: compound values of iterator-typed objects, when the
//                        iterator object is treated as an rvalue taken of a
//                        particular lvalue, eg. a copy of "type-a" iterator
//                        object, or an iterator that existed before the
//                        analysis has started.
//
// To handle any of these three different representations stored in an SVal we
// use setter and getters functions which separate the three cases. To store
// them we use a pointer union of symbol and memory region.
//
// The checker works the following way: We record the begin and the
// past-end iterator for all containers whenever their `.begin()` and `.end()`
// are called. Since the Constraint Manager cannot handle such SVals we need
// to take over its role. We post-check equality and non-equality comparisons
// and record that the two sides are equal if we are in the 'equal' branch
// (true-branch for `==` and false-branch for `!=`).
//
// In case of type-I or type-II iterators we get a concrete integer as a result
// of the comparison (1 or 0) but in case of type-III we only get a Symbol. In
// this latter case we record the symbol and reload it in evalAssume() and do
// the propagation there. We also handle (maybe double) negated comparisons
// which are represented in the form of (x == 0 or x != 0) where x is the
// comparison itself.
//
// Since `SimpleConstraintManager` cannot handle complex symbolic expressions
// we only use expressions of the format S, S+n or S-n for iterator positions
// where S is a conjured symbol and n is an unsigned concrete integer. When
// making an assumption e.g. `S1 + n == S2 + m` we store `S1 - S2 == m - n` as
// a constraint which we later retrieve when doing an actual comparison.

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicType.h"

#include "Iterator.h"

#include <utility>

using namespace clang;
using namespace ento;
using namespace iterator;

namespace {

class IteratorModeling
    : public Checker<check::PostCall, check::PostStmt<MaterializeTemporaryExpr>,
                     check::Bind, check::LiveSymbols, check::DeadSymbols> {

  void handleComparison(CheckerContext &C, const Expr *CE, SVal RetVal,
                        const SVal &LVal, const SVal &RVal,
                        OverloadedOperatorKind Op) const;
  void processComparison(CheckerContext &C, ProgramStateRef State,
                         SymbolRef Sym1, SymbolRef Sym2, const SVal &RetVal,
                         OverloadedOperatorKind Op) const;
  void handleIncrement(CheckerContext &C, const SVal &RetVal, const SVal &Iter,
                       bool Postfix) const;
  void handleDecrement(CheckerContext &C, const SVal &RetVal, const SVal &Iter,
                       bool Postfix) const;
  void handleRandomIncrOrDecr(CheckerContext &C, const Expr *CE,
                              OverloadedOperatorKind Op, const SVal &RetVal,
                              const SVal &LHS, const SVal &RHS) const;
  void handleBegin(CheckerContext &C, const Expr *CE, const SVal &RetVal,
                   const SVal &Cont) const;
  void handleEnd(CheckerContext &C, const Expr *CE, const SVal &RetVal,
                 const SVal &Cont) const;
  void assignToContainer(CheckerContext &C, const Expr *CE, const SVal &RetVal,
                         const MemRegion *Cont) const;
  void handleAssign(CheckerContext &C, const SVal &Cont,
                    const Expr *CE = nullptr,
                    const SVal &OldCont = UndefinedVal()) const;
  void handleClear(CheckerContext &C, const SVal &Cont) const;
  void handlePushBack(CheckerContext &C, const SVal &Cont) const;
  void handlePopBack(CheckerContext &C, const SVal &Cont) const;
  void handlePushFront(CheckerContext &C, const SVal &Cont) const;
  void handlePopFront(CheckerContext &C, const SVal &Cont) const;
  void handleInsert(CheckerContext &C, const SVal &Iter) const;
  void handleErase(CheckerContext &C, const SVal &Iter) const;
  void handleErase(CheckerContext &C, const SVal &Iter1,
                   const SVal &Iter2) const;
  void handleEraseAfter(CheckerContext &C, const SVal &Iter) const;
  void handleEraseAfter(CheckerContext &C, const SVal &Iter1,
                        const SVal &Iter2) const;
  void printState(raw_ostream &Out, ProgramStateRef State, const char *NL,
                  const char *Sep) const override;

public:
  IteratorModeling() {}

  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkBind(SVal Loc, SVal Val, const Stmt *S, CheckerContext &C) const;
  void checkPostStmt(const CXXConstructExpr *CCE, CheckerContext &C) const;
  void checkPostStmt(const DeclStmt *DS, CheckerContext &C) const;
  void checkPostStmt(const MaterializeTemporaryExpr *MTE,
                     CheckerContext &C) const;
  void checkLiveSymbols(ProgramStateRef State, SymbolReaper &SR) const;
  void checkDeadSymbols(SymbolReaper &SR, CheckerContext &C) const;
};

bool isBeginCall(const FunctionDecl *Func);
bool isEndCall(const FunctionDecl *Func);
bool isAssignCall(const FunctionDecl *Func);
bool isClearCall(const FunctionDecl *Func);
bool isPushBackCall(const FunctionDecl *Func);
bool isEmplaceBackCall(const FunctionDecl *Func);
bool isPopBackCall(const FunctionDecl *Func);
bool isPushFrontCall(const FunctionDecl *Func);
bool isEmplaceFrontCall(const FunctionDecl *Func);
bool isPopFrontCall(const FunctionDecl *Func);
bool isAssignmentOperator(OverloadedOperatorKind OK);
bool isSimpleComparisonOperator(OverloadedOperatorKind OK);
bool hasSubscriptOperator(ProgramStateRef State, const MemRegion *Reg);
bool frontModifiable(ProgramStateRef State, const MemRegion *Reg);
bool backModifiable(ProgramStateRef State, const MemRegion *Reg);
SymbolRef getContainerBegin(ProgramStateRef State, const MemRegion *Cont);
SymbolRef getContainerEnd(ProgramStateRef State, const MemRegion *Cont);
ProgramStateRef createContainerBegin(ProgramStateRef State,
                                     const MemRegion *Cont, const Expr *E,
                                     QualType T, const LocationContext *LCtx,
                                     unsigned BlockCount);
ProgramStateRef createContainerEnd(ProgramStateRef State, const MemRegion *Cont,
                                   const Expr *E, QualType T,
                                   const LocationContext *LCtx,
                                   unsigned BlockCount);
ProgramStateRef setContainerData(ProgramStateRef State, const MemRegion *Cont,
                                 const ContainerData &CData);
ProgramStateRef removeIteratorPosition(ProgramStateRef State, const SVal &Val);
ProgramStateRef assumeNoOverflow(ProgramStateRef State, SymbolRef Sym,
                                 long Scale);
ProgramStateRef invalidateAllIteratorPositions(ProgramStateRef State,
                                               const MemRegion *Cont);
ProgramStateRef
invalidateAllIteratorPositionsExcept(ProgramStateRef State,
                                     const MemRegion *Cont, SymbolRef Offset,
                                     BinaryOperator::Opcode Opc);
ProgramStateRef invalidateIteratorPositions(ProgramStateRef State,
                                            SymbolRef Offset,
                                            BinaryOperator::Opcode Opc);
ProgramStateRef invalidateIteratorPositions(ProgramStateRef State,
                                            SymbolRef Offset1,
                                            BinaryOperator::Opcode Opc1,
                                            SymbolRef Offset2,
                                            BinaryOperator::Opcode Opc2);
ProgramStateRef reassignAllIteratorPositions(ProgramStateRef State,
                                             const MemRegion *Cont,
                                             const MemRegion *NewCont);
ProgramStateRef reassignAllIteratorPositionsUnless(ProgramStateRef State,
                                                   const MemRegion *Cont,
                                                   const MemRegion *NewCont,
                                                   SymbolRef Offset,
                                                   BinaryOperator::Opcode Opc);
ProgramStateRef rebaseSymbolInIteratorPositionsIf(
    ProgramStateRef State, SValBuilder &SVB, SymbolRef OldSym,
    SymbolRef NewSym, SymbolRef CondSym, BinaryOperator::Opcode Opc);
ProgramStateRef relateSymbols(ProgramStateRef State, SymbolRef Sym1,
                              SymbolRef Sym2, bool Equal);
SymbolRef rebaseSymbol(ProgramStateRef State, SValBuilder &SVB, SymbolRef Expr,
                        SymbolRef OldSym, SymbolRef NewSym);
bool hasLiveIterators(ProgramStateRef State, const MemRegion *Cont);
bool isBoundThroughLazyCompoundVal(const Environment &Env,
                                   const MemRegion *Reg);

} // namespace

void IteratorModeling::checkPostCall(const CallEvent &Call,
                                     CheckerContext &C) const {
  // Record new iterator positions and iterator position changes
  const auto *Func = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!Func)
    return;

  if (Func->isOverloadedOperator()) {
    const auto Op = Func->getOverloadedOperator();
    if (isAssignmentOperator(Op)) {
      // Overloaded 'operator=' must be a non-static member function.
      const auto *InstCall = cast<CXXInstanceCall>(&Call);
      if (cast<CXXMethodDecl>(Func)->isMoveAssignmentOperator()) {
        handleAssign(C, InstCall->getCXXThisVal(), Call.getOriginExpr(),
                     Call.getArgSVal(0));
        return;
      }

      handleAssign(C, InstCall->getCXXThisVal());
      return;
    } else if (isSimpleComparisonOperator(Op)) {
      const auto *OrigExpr = Call.getOriginExpr();
      if (!OrigExpr)
        return;

      if (const auto *InstCall = dyn_cast<CXXInstanceCall>(&Call)) {
        handleComparison(C, OrigExpr, Call.getReturnValue(),
                         InstCall->getCXXThisVal(), Call.getArgSVal(0), Op);
        return;
      }

      handleComparison(C, OrigExpr, Call.getReturnValue(), Call.getArgSVal(0),
                         Call.getArgSVal(1), Op);
      return;
    } else if (isRandomIncrOrDecrOperator(Func->getOverloadedOperator())) {
      const auto *OrigExpr = Call.getOriginExpr();
      if (!OrigExpr)
        return;

      if (const auto *InstCall = dyn_cast<CXXInstanceCall>(&Call)) {
        if (Call.getNumArgs() >= 1 &&
              Call.getArgExpr(0)->getType()->isIntegralOrEnumerationType()) {
          handleRandomIncrOrDecr(C, OrigExpr, Func->getOverloadedOperator(),
                                 Call.getReturnValue(),
                                 InstCall->getCXXThisVal(), Call.getArgSVal(0));
          return;
        }
      } else {
        if (Call.getNumArgs() >= 2 &&
              Call.getArgExpr(1)->getType()->isIntegralOrEnumerationType()) {
          handleRandomIncrOrDecr(C, OrigExpr, Func->getOverloadedOperator(),
                                 Call.getReturnValue(), Call.getArgSVal(0),
                                 Call.getArgSVal(1));
          return;
        }
      }
    } else if (isIncrementOperator(Func->getOverloadedOperator())) {
      if (const auto *InstCall = dyn_cast<CXXInstanceCall>(&Call)) {
        handleIncrement(C, Call.getReturnValue(), InstCall->getCXXThisVal(),
                        Call.getNumArgs());
        return;
      }

      handleIncrement(C, Call.getReturnValue(), Call.getArgSVal(0),
                      Call.getNumArgs());
      return;
    } else if (isDecrementOperator(Func->getOverloadedOperator())) {
      if (const auto *InstCall = dyn_cast<CXXInstanceCall>(&Call)) {
        handleDecrement(C, Call.getReturnValue(), InstCall->getCXXThisVal(),
                        Call.getNumArgs());
        return;
      }

      handleDecrement(C, Call.getReturnValue(), Call.getArgSVal(0),
                        Call.getNumArgs());
      return;
    }
  } else {
    if (const auto *InstCall = dyn_cast<CXXInstanceCall>(&Call)) {
      if (isAssignCall(Func)) {
        handleAssign(C, InstCall->getCXXThisVal());
        return;
      }

      if (isClearCall(Func)) {
        handleClear(C, InstCall->getCXXThisVal());
        return;
      }

      if (isPushBackCall(Func) || isEmplaceBackCall(Func)) {
        handlePushBack(C, InstCall->getCXXThisVal());
        return;
      }

      if (isPopBackCall(Func)) {
        handlePopBack(C, InstCall->getCXXThisVal());
        return;
      }

      if (isPushFrontCall(Func) || isEmplaceFrontCall(Func)) {
        handlePushFront(C, InstCall->getCXXThisVal());
        return;
      }

      if (isPopFrontCall(Func)) {
        handlePopFront(C, InstCall->getCXXThisVal());
        return;
      }

      if (isInsertCall(Func) || isEmplaceCall(Func)) {
        handleInsert(C, Call.getArgSVal(0));
        return;
      }

      if (isEraseCall(Func)) {
        if (Call.getNumArgs() == 1) {
          handleErase(C, Call.getArgSVal(0));
          return;
        }

        if (Call.getNumArgs() == 2) {
          handleErase(C, Call.getArgSVal(0), Call.getArgSVal(1));
          return;
        }
      }

      if (isEraseAfterCall(Func)) {
        if (Call.getNumArgs() == 1) {
          handleEraseAfter(C, Call.getArgSVal(0));
          return;
        }

        if (Call.getNumArgs() == 2) {
          handleEraseAfter(C, Call.getArgSVal(0), Call.getArgSVal(1));
          return;
        }
      }
    }

    const auto *OrigExpr = Call.getOriginExpr();
    if (!OrigExpr)
      return;

    if (!isIteratorType(Call.getResultType()))
      return;

    auto State = C.getState();

    if (const auto *InstCall = dyn_cast<CXXInstanceCall>(&Call)) {
      if (isBeginCall(Func)) {
        handleBegin(C, OrigExpr, Call.getReturnValue(),
                    InstCall->getCXXThisVal());
        return;
      }
      
      if (isEndCall(Func)) {
        handleEnd(C, OrigExpr, Call.getReturnValue(),
                  InstCall->getCXXThisVal());
        return;
      }
    }

    // Already bound to container?
    if (getIteratorPosition(State, Call.getReturnValue()))
      return;

    // Copy-like and move constructors
    if (isa<CXXConstructorCall>(&Call) && Call.getNumArgs() == 1) {
      if (const auto *Pos = getIteratorPosition(State, Call.getArgSVal(0))) {
        State = setIteratorPosition(State, Call.getReturnValue(), *Pos);
        if (cast<CXXConstructorDecl>(Func)->isMoveConstructor()) {
          State = removeIteratorPosition(State, Call.getArgSVal(0));
        }
        C.addTransition(State);
        return;
      }
    }

    // Assumption: if return value is an iterator which is not yet bound to a
    //             container, then look for the first iterator argument, and
    //             bind the return value to the same container. This approach
    //             works for STL algorithms.
    // FIXME: Add a more conservative mode
    for (unsigned i = 0; i < Call.getNumArgs(); ++i) {
      if (isIteratorType(Call.getArgExpr(i)->getType())) {
        if (const auto *Pos = getIteratorPosition(State, Call.getArgSVal(i))) {
          assignToContainer(C, OrigExpr, Call.getReturnValue(),
                            Pos->getContainer());
          return;
        }
      }
    }
  }
}

void IteratorModeling::checkBind(SVal Loc, SVal Val, const Stmt *S,
                                 CheckerContext &C) const {
  auto State = C.getState();
  const auto *Pos = getIteratorPosition(State, Val);
  if (Pos) {
    State = setIteratorPosition(State, Loc, *Pos);
    C.addTransition(State);
  } else {
    const auto *OldPos = getIteratorPosition(State, Loc);
    if (OldPos) {
      State = removeIteratorPosition(State, Loc);
      C.addTransition(State);
    }
  }
}

void IteratorModeling::checkPostStmt(const MaterializeTemporaryExpr *MTE,
                                     CheckerContext &C) const {
  /* Transfer iterator state to temporary objects */
  auto State = C.getState();
  const auto *Pos = getIteratorPosition(State, C.getSVal(MTE->getSubExpr()));
  if (!Pos)
    return;
  State = setIteratorPosition(State, C.getSVal(MTE), *Pos);
  C.addTransition(State);
}

void IteratorModeling::checkLiveSymbols(ProgramStateRef State,
                                        SymbolReaper &SR) const {
  // Keep symbolic expressions of iterator positions, container begins and ends
  // alive
  auto RegionMap = State->get<IteratorRegionMap>();
  for (const auto &Reg : RegionMap) {
    const auto Offset = Reg.second.getOffset();
    for (auto i = Offset->symbol_begin(); i != Offset->symbol_end(); ++i)
      if (isa<SymbolData>(*i))
        SR.markLive(*i);
  }

  auto SymbolMap = State->get<IteratorSymbolMap>();
  for (const auto &Sym : SymbolMap) {
    const auto Offset = Sym.second.getOffset();
    for (auto i = Offset->symbol_begin(); i != Offset->symbol_end(); ++i)
      if (isa<SymbolData>(*i))
        SR.markLive(*i);
  }

  auto ContMap = State->get<ContainerMap>();
  for (const auto &Cont : ContMap) {
    const auto CData = Cont.second;
    if (CData.getBegin()) {
      SR.markLive(CData.getBegin());
      if(const auto *SIE = dyn_cast<SymIntExpr>(CData.getBegin()))
        SR.markLive(SIE->getLHS());
    }
    if (CData.getEnd()) {
      SR.markLive(CData.getEnd());
      if(const auto *SIE = dyn_cast<SymIntExpr>(CData.getEnd()))
        SR.markLive(SIE->getLHS());
    }
  }
}

void IteratorModeling::checkDeadSymbols(SymbolReaper &SR,
                                        CheckerContext &C) const {
  // Cleanup
  auto State = C.getState();

  auto RegionMap = State->get<IteratorRegionMap>();
  for (const auto &Reg : RegionMap) {
    if (!SR.isLiveRegion(Reg.first)) {
      // The region behind the `LazyCompoundVal` is often cleaned up before
      // the `LazyCompoundVal` itself. If there are iterator positions keyed
      // by these regions their cleanup must be deferred.
      if (!isBoundThroughLazyCompoundVal(State->getEnvironment(), Reg.first)) {
        State = State->remove<IteratorRegionMap>(Reg.first);
      }
    }
  }

  auto SymbolMap = State->get<IteratorSymbolMap>();
  for (const auto &Sym : SymbolMap) {
    if (!SR.isLive(Sym.first)) {
      State = State->remove<IteratorSymbolMap>(Sym.first);
    }
  }

  auto ContMap = State->get<ContainerMap>();
  for (const auto &Cont : ContMap) {
    if (!SR.isLiveRegion(Cont.first)) {
      // We must keep the container data while it has live iterators to be able
      // to compare them to the begin and the end of the container.
      if (!hasLiveIterators(State, Cont.first)) {
        State = State->remove<ContainerMap>(Cont.first);
      }
    }
  }

  C.addTransition(State);
}

void IteratorModeling::handleComparison(CheckerContext &C, const Expr *CE,
                                       SVal RetVal, const SVal &LVal,
                                       const SVal &RVal,
                                       OverloadedOperatorKind Op) const {
  // Record the operands and the operator of the comparison for the next
  // evalAssume, if the result is a symbolic expression. If it is a concrete
  // value (only one branch is possible), then transfer the state between
  // the operands according to the operator and the result
   auto State = C.getState();
  const auto *LPos = getIteratorPosition(State, LVal);
  const auto *RPos = getIteratorPosition(State, RVal);
  const MemRegion *Cont = nullptr;
  if (LPos) {
    Cont = LPos->getContainer();
  } else if (RPos) {
    Cont = RPos->getContainer();
  }
  if (!Cont)
    return;

  // At least one of the iterators have recorded positions. If one of them has
  // not then create a new symbol for the offset.
  SymbolRef Sym;
  if (!LPos || !RPos) {
    auto &SymMgr = C.getSymbolManager();
    Sym = SymMgr.conjureSymbol(CE, C.getLocationContext(),
                               C.getASTContext().LongTy, C.blockCount());
    State = assumeNoOverflow(State, Sym, 4);
  }

  if (!LPos) {
    State = setIteratorPosition(State, LVal,
                                IteratorPosition::getPosition(Cont, Sym));
    LPos = getIteratorPosition(State, LVal);
  } else if (!RPos) {
    State = setIteratorPosition(State, RVal,
                                IteratorPosition::getPosition(Cont, Sym));
    RPos = getIteratorPosition(State, RVal);
  }

  // We cannot make assumpotions on `UnknownVal`. Let us conjure a symbol
  // instead.
  if (RetVal.isUnknown()) {
    auto &SymMgr = C.getSymbolManager();
    auto *LCtx = C.getLocationContext();
    RetVal = nonloc::SymbolVal(SymMgr.conjureSymbol(
        CE, LCtx, C.getASTContext().BoolTy, C.blockCount()));
    State = State->BindExpr(CE, LCtx, RetVal);
  }

  processComparison(C, State, LPos->getOffset(), RPos->getOffset(), RetVal, Op);
}

void IteratorModeling::processComparison(CheckerContext &C,
                                         ProgramStateRef State, SymbolRef Sym1,
                                         SymbolRef Sym2, const SVal &RetVal,
                                         OverloadedOperatorKind Op) const {
  if (const auto TruthVal = RetVal.getAs<nonloc::ConcreteInt>()) {
    if ((State = relateSymbols(State, Sym1, Sym2,
                              (Op == OO_EqualEqual) ==
                               (TruthVal->getValue() != 0)))) {
      C.addTransition(State);
    } else {
      C.generateSink(State, C.getPredecessor());
    }
    return;
  }

  const auto ConditionVal = RetVal.getAs<DefinedSVal>();
  if (!ConditionVal)
    return;

  if (auto StateTrue = relateSymbols(State, Sym1, Sym2, Op == OO_EqualEqual)) {
    StateTrue = StateTrue->assume(*ConditionVal, true);
    C.addTransition(StateTrue);
  }
  
  if (auto StateFalse = relateSymbols(State, Sym1, Sym2, Op != OO_EqualEqual)) {
    StateFalse = StateFalse->assume(*ConditionVal, false);
    C.addTransition(StateFalse);
  }
}

void IteratorModeling::handleIncrement(CheckerContext &C, const SVal &RetVal,
                                       const SVal &Iter, bool Postfix) const {
  // Increment the symbolic expressions which represents the position of the
  // iterator
  auto State = C.getState();
  auto &BVF = C.getSymbolManager().getBasicVals();

  const auto *Pos = getIteratorPosition(State, Iter);
  if (!Pos)
    return;

  auto NewState =
    advancePosition(State, Iter, OO_Plus,
                    nonloc::ConcreteInt(BVF.getValue(llvm::APSInt::get(1))));
  assert(NewState &&
         "Advancing position by concrete int should always be successful");

  const auto *NewPos = getIteratorPosition(NewState, Iter);
  assert(NewPos &&
         "Iterator should have position after successful advancement");

  State = setIteratorPosition(State, Iter, *NewPos);
  State = setIteratorPosition(State, RetVal, Postfix ? *Pos : *NewPos);
  C.addTransition(State);
}

void IteratorModeling::handleDecrement(CheckerContext &C, const SVal &RetVal,
                                       const SVal &Iter, bool Postfix) const {
  // Decrement the symbolic expressions which represents the position of the
  // iterator
  auto State = C.getState();
  auto &BVF = C.getSymbolManager().getBasicVals();

  const auto *Pos = getIteratorPosition(State, Iter);
  if (!Pos)
    return;

  auto NewState =
    advancePosition(State, Iter, OO_Minus,
                    nonloc::ConcreteInt(BVF.getValue(llvm::APSInt::get(1))));
  assert(NewState &&
         "Advancing position by concrete int should always be successful");

  const auto *NewPos = getIteratorPosition(NewState, Iter);
  assert(NewPos &&
         "Iterator should have position after successful advancement");

  State = setIteratorPosition(State, Iter, *NewPos);
  State = setIteratorPosition(State, RetVal, Postfix ? *Pos : *NewPos);
  C.addTransition(State);
}

void IteratorModeling::handleRandomIncrOrDecr(CheckerContext &C,
                                              const Expr *CE,
                                              OverloadedOperatorKind Op,
                                              const SVal &RetVal,
                                              const SVal &LHS,
                                              const SVal &RHS) const {
  // Increment or decrement the symbolic expressions which represents the
  // position of the iterator
  auto State = C.getState();

  const auto *Pos = getIteratorPosition(State, LHS);
  if (!Pos)
    return;

  const auto *value = &RHS;
  if (auto loc = RHS.getAs<Loc>()) {
    const auto val = State->getRawSVal(*loc);
    value = &val;
  }

  auto &TgtVal = (Op == OO_PlusEqual || Op == OO_MinusEqual) ? LHS : RetVal;

  auto NewState =
    advancePosition(State, LHS, Op, *value);
  if (NewState) {
    const auto *NewPos = getIteratorPosition(NewState, LHS);
    assert(NewPos &&
           "Iterator should have position after successful advancement");

    State = setIteratorPosition(NewState, TgtVal, *NewPos);
    C.addTransition(State);
  } else {
    assignToContainer(C, CE, TgtVal, Pos->getContainer());
  }
}

void IteratorModeling::handleBegin(CheckerContext &C, const Expr *CE,
                                   const SVal &RetVal, const SVal &Cont) const {
  const auto *ContReg = Cont.getAsRegion();
  if (!ContReg)
    return;

  ContReg = ContReg->getMostDerivedObjectRegion();

  // If the container already has a begin symbol then use it. Otherwise first
  // create a new one.
  auto State = C.getState();
  auto BeginSym = getContainerBegin(State, ContReg);
  if (!BeginSym) {
    State = createContainerBegin(State, ContReg, CE, C.getASTContext().LongTy,
                                 C.getLocationContext(), C.blockCount());
    BeginSym = getContainerBegin(State, ContReg);
  }
  State = setIteratorPosition(State, RetVal,
                              IteratorPosition::getPosition(ContReg, BeginSym));
  C.addTransition(State);
}

void IteratorModeling::handleEnd(CheckerContext &C, const Expr *CE,
                                 const SVal &RetVal, const SVal &Cont) const {
  const auto *ContReg = Cont.getAsRegion();
  if (!ContReg)
    return;

  ContReg = ContReg->getMostDerivedObjectRegion();

  // If the container already has an end symbol then use it. Otherwise first
  // create a new one.
  auto State = C.getState();
  auto EndSym = getContainerEnd(State, ContReg);
  if (!EndSym) {
    State = createContainerEnd(State, ContReg, CE, C.getASTContext().LongTy,
                               C.getLocationContext(), C.blockCount());
    EndSym = getContainerEnd(State, ContReg);
  }
  State = setIteratorPosition(State, RetVal,
                              IteratorPosition::getPosition(ContReg, EndSym));
  C.addTransition(State);
}

void IteratorModeling::assignToContainer(CheckerContext &C, const Expr *CE,
                                         const SVal &RetVal,
                                         const MemRegion *Cont) const {
  Cont = Cont->getMostDerivedObjectRegion();

  auto State = C.getState();
  auto &SymMgr = C.getSymbolManager();
  auto Sym = SymMgr.conjureSymbol(CE, C.getLocationContext(),
                                  C.getASTContext().LongTy, C.blockCount());
  State = assumeNoOverflow(State, Sym, 4);
  State = setIteratorPosition(State, RetVal,
                              IteratorPosition::getPosition(Cont, Sym));
  C.addTransition(State);
}

void IteratorModeling::handleAssign(CheckerContext &C, const SVal &Cont,
                                    const Expr *CE, const SVal &OldCont) const {
  const auto *ContReg = Cont.getAsRegion();
  if (!ContReg)
    return;

  ContReg = ContReg->getMostDerivedObjectRegion();

  // Assignment of a new value to a container always invalidates all its
  // iterators
  auto State = C.getState();
  const auto CData = getContainerData(State, ContReg);
  if (CData) {
    State = invalidateAllIteratorPositions(State, ContReg);
  }

  // In case of move, iterators of the old container (except the past-end
  // iterators) remain valid but refer to the new container
  if (!OldCont.isUndef()) {
    const auto *OldContReg = OldCont.getAsRegion();
    if (OldContReg) {
      OldContReg = OldContReg->getMostDerivedObjectRegion();
      const auto OldCData = getContainerData(State, OldContReg);
      if (OldCData) {
        if (const auto OldEndSym = OldCData->getEnd()) {
          // If we already assigned an "end" symbol to the old container, then
          // first reassign all iterator positions to the new container which
          // are not past the container (thus not greater or equal to the
          // current "end" symbol).
          State = reassignAllIteratorPositionsUnless(State, OldContReg, ContReg,
                                                     OldEndSym, BO_GE);
          auto &SymMgr = C.getSymbolManager();
          auto &SVB = C.getSValBuilder();
          // Then generate and assign a new "end" symbol for the new container.
          auto NewEndSym =
              SymMgr.conjureSymbol(CE, C.getLocationContext(),
                                   C.getASTContext().LongTy, C.blockCount());
          State = assumeNoOverflow(State, NewEndSym, 4);
          if (CData) {
            State = setContainerData(State, ContReg, CData->newEnd(NewEndSym));
          } else {
            State = setContainerData(State, ContReg,
                                     ContainerData::fromEnd(NewEndSym));
          }
          // Finally, replace the old "end" symbol in the already reassigned
          // iterator positions with the new "end" symbol.
          State = rebaseSymbolInIteratorPositionsIf(
              State, SVB, OldEndSym, NewEndSym, OldEndSym, BO_LT);
        } else {
          // There was no "end" symbol assigned yet to the old container,
          // so reassign all iterator positions to the new container.
          State = reassignAllIteratorPositions(State, OldContReg, ContReg);
        }
        if (const auto OldBeginSym = OldCData->getBegin()) {
          // If we already assigned a "begin" symbol to the old container, then
          // assign it to the new container and remove it from the old one.
          if (CData) {
            State =
                setContainerData(State, ContReg, CData->newBegin(OldBeginSym));
          } else {
            State = setContainerData(State, ContReg,
                                     ContainerData::fromBegin(OldBeginSym));
          }
          State =
              setContainerData(State, OldContReg, OldCData->newEnd(nullptr));
        }
      } else {
        // There was neither "begin" nor "end" symbol assigned yet to the old
        // container, so reassign all iterator positions to the new container.
        State = reassignAllIteratorPositions(State, OldContReg, ContReg);
      }
    }
  }
  C.addTransition(State);
}

void IteratorModeling::handleClear(CheckerContext &C, const SVal &Cont) const {
  const auto *ContReg = Cont.getAsRegion();
  if (!ContReg)
    return;

  ContReg = ContReg->getMostDerivedObjectRegion();

  // The clear() operation invalidates all the iterators, except the past-end
  // iterators of list-like containers
  auto State = C.getState();
  if (!hasSubscriptOperator(State, ContReg) ||
      !backModifiable(State, ContReg)) {
    const auto CData = getContainerData(State, ContReg);
    if (CData) {
      if (const auto EndSym = CData->getEnd()) {
        State =
            invalidateAllIteratorPositionsExcept(State, ContReg, EndSym, BO_GE);
        C.addTransition(State);
        return;
      }
    }
  }
  State = invalidateAllIteratorPositions(State, ContReg);
  C.addTransition(State);
}

void IteratorModeling::handlePushBack(CheckerContext &C,
                                      const SVal &Cont) const {
  const auto *ContReg = Cont.getAsRegion();
  if (!ContReg)
    return;

  ContReg = ContReg->getMostDerivedObjectRegion();

  // For deque-like containers invalidate all iterator positions
  auto State = C.getState();
  if (hasSubscriptOperator(State, ContReg) && frontModifiable(State, ContReg)) {
    State = invalidateAllIteratorPositions(State, ContReg);
    C.addTransition(State);
    return;
  }

  const auto CData = getContainerData(State, ContReg);
  if (!CData)
    return;

  // For vector-like containers invalidate the past-end iterator positions
  if (const auto EndSym = CData->getEnd()) {
    if (hasSubscriptOperator(State, ContReg)) {
      State = invalidateIteratorPositions(State, EndSym, BO_GE);
    }
    auto &SymMgr = C.getSymbolManager();
    auto &BVF = SymMgr.getBasicVals();
    auto &SVB = C.getSValBuilder();
    const auto newEndSym =
      SVB.evalBinOp(State, BO_Add,
                    nonloc::SymbolVal(EndSym),
                    nonloc::ConcreteInt(BVF.getValue(llvm::APSInt::get(1))),
                    SymMgr.getType(EndSym)).getAsSymbol();
    State = setContainerData(State, ContReg, CData->newEnd(newEndSym));
  }
  C.addTransition(State);
}

void IteratorModeling::handlePopBack(CheckerContext &C,
                                     const SVal &Cont) const {
  const auto *ContReg = Cont.getAsRegion();
  if (!ContReg)
    return;

  ContReg = ContReg->getMostDerivedObjectRegion();

  auto State = C.getState();
  const auto CData = getContainerData(State, ContReg);
  if (!CData)
    return;

  if (const auto EndSym = CData->getEnd()) {
    auto &SymMgr = C.getSymbolManager();
    auto &BVF = SymMgr.getBasicVals();
    auto &SVB = C.getSValBuilder();
    const auto BackSym =
      SVB.evalBinOp(State, BO_Sub,
                    nonloc::SymbolVal(EndSym),
                    nonloc::ConcreteInt(BVF.getValue(llvm::APSInt::get(1))),
                    SymMgr.getType(EndSym)).getAsSymbol();
    // For vector-like and deque-like containers invalidate the last and the
    // past-end iterator positions. For list-like containers only invalidate
    // the last position
    if (hasSubscriptOperator(State, ContReg) &&
        backModifiable(State, ContReg)) {
      State = invalidateIteratorPositions(State, BackSym, BO_GE);
      State = setContainerData(State, ContReg, CData->newEnd(nullptr));
    } else {
      State = invalidateIteratorPositions(State, BackSym, BO_EQ);
    }
    auto newEndSym = BackSym;
    State = setContainerData(State, ContReg, CData->newEnd(newEndSym));
    C.addTransition(State);
  }
}

void IteratorModeling::handlePushFront(CheckerContext &C,
                                       const SVal &Cont) const {
  const auto *ContReg = Cont.getAsRegion();
  if (!ContReg)
    return;

  ContReg = ContReg->getMostDerivedObjectRegion();

  // For deque-like containers invalidate all iterator positions
  auto State = C.getState();
  if (hasSubscriptOperator(State, ContReg)) {
    State = invalidateAllIteratorPositions(State, ContReg);
    C.addTransition(State);
  } else {
    const auto CData = getContainerData(State, ContReg);
    if (!CData)
      return;

    if (const auto BeginSym = CData->getBegin()) {
      auto &SymMgr = C.getSymbolManager();
      auto &BVF = SymMgr.getBasicVals();
      auto &SVB = C.getSValBuilder();
      const auto newBeginSym =
        SVB.evalBinOp(State, BO_Sub,
                      nonloc::SymbolVal(BeginSym),
                      nonloc::ConcreteInt(BVF.getValue(llvm::APSInt::get(1))),
                      SymMgr.getType(BeginSym)).getAsSymbol();
      State = setContainerData(State, ContReg, CData->newBegin(newBeginSym));
      C.addTransition(State);
    }
  }
}

void IteratorModeling::handlePopFront(CheckerContext &C,
                                      const SVal &Cont) const {
  const auto *ContReg = Cont.getAsRegion();
  if (!ContReg)
    return;

  ContReg = ContReg->getMostDerivedObjectRegion();

  auto State = C.getState();
  const auto CData = getContainerData(State, ContReg);
  if (!CData)
    return;

  // For deque-like containers invalidate all iterator positions. For list-like
  // iterators only invalidate the first position
  if (const auto BeginSym = CData->getBegin()) {
    if (hasSubscriptOperator(State, ContReg)) {
      State = invalidateIteratorPositions(State, BeginSym, BO_LE);
    } else {
      State = invalidateIteratorPositions(State, BeginSym, BO_EQ);
    }
    auto &SymMgr = C.getSymbolManager();
    auto &BVF = SymMgr.getBasicVals();
    auto &SVB = C.getSValBuilder();
    const auto newBeginSym =
      SVB.evalBinOp(State, BO_Add,
                    nonloc::SymbolVal(BeginSym),
                    nonloc::ConcreteInt(BVF.getValue(llvm::APSInt::get(1))),
                    SymMgr.getType(BeginSym)).getAsSymbol();
    State = setContainerData(State, ContReg, CData->newBegin(newBeginSym));
    C.addTransition(State);
  }
}

void IteratorModeling::handleInsert(CheckerContext &C, const SVal &Iter) const {
  auto State = C.getState();
  const auto *Pos = getIteratorPosition(State, Iter);
  if (!Pos)
    return;

  // For deque-like containers invalidate all iterator positions. For
  // vector-like containers invalidate iterator positions after the insertion.
  const auto *Cont = Pos->getContainer();
  if (hasSubscriptOperator(State, Cont) && backModifiable(State, Cont)) {
    if (frontModifiable(State, Cont)) {
      State = invalidateAllIteratorPositions(State, Cont);
    } else {
      State = invalidateIteratorPositions(State, Pos->getOffset(), BO_GE);
    }
    if (const auto *CData = getContainerData(State, Cont)) {
      if (const auto EndSym = CData->getEnd()) {
        State = invalidateIteratorPositions(State, EndSym, BO_GE);
        State = setContainerData(State, Cont, CData->newEnd(nullptr));
      }
    }
    C.addTransition(State);
  }
}

void IteratorModeling::handleErase(CheckerContext &C, const SVal &Iter) const {
  auto State = C.getState();
  const auto *Pos = getIteratorPosition(State, Iter);
  if (!Pos)
    return;

  // For deque-like containers invalidate all iterator positions. For
  // vector-like containers invalidate iterator positions at and after the
  // deletion. For list-like containers only invalidate the deleted position.
  const auto *Cont = Pos->getContainer();
  if (hasSubscriptOperator(State, Cont) && backModifiable(State, Cont)) {
    if (frontModifiable(State, Cont)) {
      State = invalidateAllIteratorPositions(State, Cont);
    } else {
      State = invalidateIteratorPositions(State, Pos->getOffset(), BO_GE);
    }
    if (const auto *CData = getContainerData(State, Cont)) {
      if (const auto EndSym = CData->getEnd()) {
        State = invalidateIteratorPositions(State, EndSym, BO_GE);
        State = setContainerData(State, Cont, CData->newEnd(nullptr));
      }
    }
  } else {
    State = invalidateIteratorPositions(State, Pos->getOffset(), BO_EQ);
  }
  C.addTransition(State);
}

void IteratorModeling::handleErase(CheckerContext &C, const SVal &Iter1,
                                   const SVal &Iter2) const {
  auto State = C.getState();
  const auto *Pos1 = getIteratorPosition(State, Iter1);
  const auto *Pos2 = getIteratorPosition(State, Iter2);
  if (!Pos1 || !Pos2)
    return;

  // For deque-like containers invalidate all iterator positions. For
  // vector-like containers invalidate iterator positions at and after the
  // deletion range. For list-like containers only invalidate the deleted
  // position range [first..last].
  const auto *Cont = Pos1->getContainer();
  if (hasSubscriptOperator(State, Cont) && backModifiable(State, Cont)) {
    if (frontModifiable(State, Cont)) {
      State = invalidateAllIteratorPositions(State, Cont);
    } else {
      State = invalidateIteratorPositions(State, Pos1->getOffset(), BO_GE);
    }
    if (const auto *CData = getContainerData(State, Cont)) {
      if (const auto EndSym = CData->getEnd()) {
        State = invalidateIteratorPositions(State, EndSym, BO_GE);
        State = setContainerData(State, Cont, CData->newEnd(nullptr));
      }
    }
  } else {
    State = invalidateIteratorPositions(State, Pos1->getOffset(), BO_GE,
                                        Pos2->getOffset(), BO_LT);
  }
  C.addTransition(State);
}

void IteratorModeling::handleEraseAfter(CheckerContext &C,
                                        const SVal &Iter) const {
  auto State = C.getState();
  const auto *Pos = getIteratorPosition(State, Iter);
  if (!Pos)
    return;

  // Invalidate the deleted iterator position, which is the position of the
  // parameter plus one.
  auto &SymMgr = C.getSymbolManager();
  auto &BVF = SymMgr.getBasicVals();
  auto &SVB = C.getSValBuilder();
  const auto NextSym =
    SVB.evalBinOp(State, BO_Add,
                  nonloc::SymbolVal(Pos->getOffset()),
                  nonloc::ConcreteInt(BVF.getValue(llvm::APSInt::get(1))),
                  SymMgr.getType(Pos->getOffset())).getAsSymbol();
  State = invalidateIteratorPositions(State, NextSym, BO_EQ);
  C.addTransition(State);
}

void IteratorModeling::handleEraseAfter(CheckerContext &C, const SVal &Iter1,
                                        const SVal &Iter2) const {
  auto State = C.getState();
  const auto *Pos1 = getIteratorPosition(State, Iter1);
  const auto *Pos2 = getIteratorPosition(State, Iter2);
  if (!Pos1 || !Pos2)
    return;

  // Invalidate the deleted iterator position range (first..last)
  State = invalidateIteratorPositions(State, Pos1->getOffset(), BO_GT,
                                      Pos2->getOffset(), BO_LT);
  C.addTransition(State);
}

void IteratorModeling::printState(raw_ostream &Out, ProgramStateRef State,
                                  const char *NL, const char *Sep) const {

  auto ContMap = State->get<ContainerMap>();

  if (!ContMap.isEmpty()) {
    Out << Sep << "Container Data :" << NL;
    for (const auto &Cont : ContMap) {
      Cont.first->dumpToStream(Out);
      Out << " : [ ";
      const auto CData = Cont.second;
      if (CData.getBegin())
        CData.getBegin()->dumpToStream(Out);
      else
        Out << "<Unknown>";
      Out << " .. ";
      if (CData.getEnd())
        CData.getEnd()->dumpToStream(Out);
      else
        Out << "<Unknown>";
      Out << " ]" << NL;
    }
  }

  auto SymbolMap = State->get<IteratorSymbolMap>();
  auto RegionMap = State->get<IteratorRegionMap>();

  if (!SymbolMap.isEmpty() || !RegionMap.isEmpty()) {
    Out << Sep << "Iterator Positions :" << NL;
    for (const auto &Sym : SymbolMap) {
      Sym.first->dumpToStream(Out);
      Out << " : ";
      const auto Pos = Sym.second;
      Out << (Pos.isValid() ? "Valid" : "Invalid") << " ; Container == ";
      Pos.getContainer()->dumpToStream(Out);
      Out<<" ; Offset == ";
      Pos.getOffset()->dumpToStream(Out);
    }

    for (const auto &Reg : RegionMap) {
      Reg.first->dumpToStream(Out);
      Out << " : ";
      const auto Pos = Reg.second;
      Out << (Pos.isValid() ? "Valid" : "Invalid") << " ; Container == ";
      Pos.getContainer()->dumpToStream(Out);
      Out<<" ; Offset == ";
      Pos.getOffset()->dumpToStream(Out);
    }
  }
}


namespace {

const CXXRecordDecl *getCXXRecordDecl(ProgramStateRef State,
                                      const MemRegion *Reg);

bool isBeginCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  return IdInfo->getName().endswith_lower("begin");
}

bool isEndCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  return IdInfo->getName().endswith_lower("end");
}

bool isAssignCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  if (Func->getNumParams() > 2)
    return false;
  return IdInfo->getName() == "assign";
}

bool isClearCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  if (Func->getNumParams() > 0)
    return false;
  return IdInfo->getName() == "clear";
}

bool isPushBackCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  if (Func->getNumParams() != 1)
    return false;
  return IdInfo->getName() == "push_back";
}

bool isEmplaceBackCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  if (Func->getNumParams() < 1)
    return false;
  return IdInfo->getName() == "emplace_back";
}

bool isPopBackCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  if (Func->getNumParams() > 0)
    return false;
  return IdInfo->getName() == "pop_back";
}

bool isPushFrontCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  if (Func->getNumParams() != 1)
    return false;
  return IdInfo->getName() == "push_front";
}

bool isEmplaceFrontCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  if (Func->getNumParams() < 1)
    return false;
  return IdInfo->getName() == "emplace_front";
}

bool isPopFrontCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  if (Func->getNumParams() > 0)
    return false;
  return IdInfo->getName() == "pop_front";
}

bool isAssignmentOperator(OverloadedOperatorKind OK) { return OK == OO_Equal; }

bool isSimpleComparisonOperator(OverloadedOperatorKind OK) {
  return OK == OO_EqualEqual || OK == OO_ExclaimEqual;
}

bool hasSubscriptOperator(ProgramStateRef State, const MemRegion *Reg) {
  const auto *CRD = getCXXRecordDecl(State, Reg);
  if (!CRD)
    return false;

  for (const auto *Method : CRD->methods()) {
    if (!Method->isOverloadedOperator())
      continue;
    const auto OPK = Method->getOverloadedOperator();
    if (OPK == OO_Subscript) {
      return true;
    }
  }
  return false;
}

bool frontModifiable(ProgramStateRef State, const MemRegion *Reg) {
  const auto *CRD = getCXXRecordDecl(State, Reg);
  if (!CRD)
    return false;

  for (const auto *Method : CRD->methods()) {
    if (!Method->getDeclName().isIdentifier())
      continue;
    if (Method->getName() == "push_front" || Method->getName() == "pop_front") {
      return true;
    }
  }
  return false;
}

bool backModifiable(ProgramStateRef State, const MemRegion *Reg) {
  const auto *CRD = getCXXRecordDecl(State, Reg);
  if (!CRD)
    return false;

  for (const auto *Method : CRD->methods()) {
    if (!Method->getDeclName().isIdentifier())
      continue;
    if (Method->getName() == "push_back" || Method->getName() == "pop_back") {
      return true;
    }
  }
  return false;
}

const CXXRecordDecl *getCXXRecordDecl(ProgramStateRef State,
                                      const MemRegion *Reg) {
  auto TI = getDynamicTypeInfo(State, Reg);
  if (!TI.isValid())
    return nullptr;

  auto Type = TI.getType();
  if (const auto *RefT = Type->getAs<ReferenceType>()) {
    Type = RefT->getPointeeType();
  }

  return Type->getUnqualifiedDesugaredType()->getAsCXXRecordDecl();
}

SymbolRef getContainerBegin(ProgramStateRef State, const MemRegion *Cont) {
  const auto *CDataPtr = getContainerData(State, Cont);
  if (!CDataPtr)
    return nullptr;

  return CDataPtr->getBegin();
}

SymbolRef getContainerEnd(ProgramStateRef State, const MemRegion *Cont) {
  const auto *CDataPtr = getContainerData(State, Cont);
  if (!CDataPtr)
    return nullptr;

  return CDataPtr->getEnd();
}

ProgramStateRef createContainerBegin(ProgramStateRef State,
                                     const MemRegion *Cont, const Expr *E,
                                     QualType T, const LocationContext *LCtx,
                                     unsigned BlockCount) {
  // Only create if it does not exist
  const auto *CDataPtr = getContainerData(State, Cont);
  if (CDataPtr && CDataPtr->getBegin())
    return State;

  auto &SymMgr = State->getSymbolManager();
  const SymbolConjured *Sym = SymMgr.conjureSymbol(E, LCtx, T, BlockCount,
                                                   "begin");
  State = assumeNoOverflow(State, Sym, 4);

  if (CDataPtr) {
    const auto CData = CDataPtr->newBegin(Sym);
    return setContainerData(State, Cont, CData);
  }

  const auto CData = ContainerData::fromBegin(Sym);
  return setContainerData(State, Cont, CData);
}

ProgramStateRef createContainerEnd(ProgramStateRef State, const MemRegion *Cont,
                                   const Expr *E, QualType T,
                                   const LocationContext *LCtx,
                                   unsigned BlockCount) {
  // Only create if it does not exist
  const auto *CDataPtr = getContainerData(State, Cont);
  if (CDataPtr && CDataPtr->getEnd())
    return State;

  auto &SymMgr = State->getSymbolManager();
  const SymbolConjured *Sym = SymMgr.conjureSymbol(E, LCtx, T, BlockCount,
                                                  "end");
  State = assumeNoOverflow(State, Sym, 4);

  if (CDataPtr) {
    const auto CData = CDataPtr->newEnd(Sym);
    return setContainerData(State, Cont, CData);
  }

  const auto CData = ContainerData::fromEnd(Sym);
  return setContainerData(State, Cont, CData);
}

ProgramStateRef setContainerData(ProgramStateRef State, const MemRegion *Cont,
                                 const ContainerData &CData) {
  return State->set<ContainerMap>(Cont, CData);
}

ProgramStateRef removeIteratorPosition(ProgramStateRef State, const SVal &Val) {
  if (auto Reg = Val.getAsRegion()) {
    Reg = Reg->getMostDerivedObjectRegion();
    return State->remove<IteratorRegionMap>(Reg);
  } else if (const auto Sym = Val.getAsSymbol()) {
    return State->remove<IteratorSymbolMap>(Sym);
  } else if (const auto LCVal = Val.getAs<nonloc::LazyCompoundVal>()) {
    return State->remove<IteratorRegionMap>(LCVal->getRegion());
  }
  return nullptr;
}

// This function tells the analyzer's engine that symbols produced by our
// checker, most notably iterator positions, are relatively small.
// A distance between items in the container should not be very large.
// By assuming that it is within around 1/8 of the address space,
// we can help the analyzer perform operations on these symbols
// without being afraid of integer overflows.
// FIXME: Should we provide it as an API, so that all checkers could use it?
ProgramStateRef assumeNoOverflow(ProgramStateRef State, SymbolRef Sym,
                                 long Scale) {
  SValBuilder &SVB = State->getStateManager().getSValBuilder();
  BasicValueFactory &BV = SVB.getBasicValueFactory();

  QualType T = Sym->getType();
  assert(T->isSignedIntegerOrEnumerationType());
  APSIntType AT = BV.getAPSIntType(T);

  ProgramStateRef NewState = State;

  llvm::APSInt Max = AT.getMaxValue() / AT.getValue(Scale);
  SVal IsCappedFromAbove =
      SVB.evalBinOpNN(State, BO_LE, nonloc::SymbolVal(Sym),
                      nonloc::ConcreteInt(Max), SVB.getConditionType());
  if (auto DV = IsCappedFromAbove.getAs<DefinedSVal>()) {
    NewState = NewState->assume(*DV, true);
    if (!NewState)
      return State;
  }

  llvm::APSInt Min = -Max;
  SVal IsCappedFromBelow =
      SVB.evalBinOpNN(State, BO_GE, nonloc::SymbolVal(Sym),
                      nonloc::ConcreteInt(Min), SVB.getConditionType());
  if (auto DV = IsCappedFromBelow.getAs<DefinedSVal>()) {
    NewState = NewState->assume(*DV, true);
    if (!NewState)
      return State;
  }

  return NewState;
}

ProgramStateRef relateSymbols(ProgramStateRef State, SymbolRef Sym1,
                              SymbolRef Sym2, bool Equal) {
  auto &SVB = State->getStateManager().getSValBuilder();

  // FIXME: This code should be reworked as follows:
  // 1. Subtract the operands using evalBinOp().
  // 2. Assume that the result doesn't overflow.
  // 3. Compare the result to 0.
  // 4. Assume the result of the comparison.
  const auto comparison =
    SVB.evalBinOp(State, BO_EQ, nonloc::SymbolVal(Sym1),
                  nonloc::SymbolVal(Sym2), SVB.getConditionType());

  assert(comparison.getAs<DefinedSVal>() &&
    "Symbol comparison must be a `DefinedSVal`");

  auto NewState = State->assume(comparison.castAs<DefinedSVal>(), Equal);
  if (!NewState)
    return nullptr;

  if (const auto CompSym = comparison.getAsSymbol()) {
    assert(isa<SymIntExpr>(CompSym) &&
           "Symbol comparison must be a `SymIntExpr`");
    assert(BinaryOperator::isComparisonOp(
               cast<SymIntExpr>(CompSym)->getOpcode()) &&
           "Symbol comparison must be a comparison");
    return assumeNoOverflow(NewState, cast<SymIntExpr>(CompSym)->getLHS(), 2);
  }

  return NewState;
}

bool hasLiveIterators(ProgramStateRef State, const MemRegion *Cont) {
  auto RegionMap = State->get<IteratorRegionMap>();
  for (const auto &Reg : RegionMap) {
    if (Reg.second.getContainer() == Cont)
      return true;
  }

  auto SymbolMap = State->get<IteratorSymbolMap>();
  for (const auto &Sym : SymbolMap) {
    if (Sym.second.getContainer() == Cont)
      return true;
  }

  return false;
}

bool isBoundThroughLazyCompoundVal(const Environment &Env,
                                   const MemRegion *Reg) {
  for (const auto &Binding : Env) {
    if (const auto LCVal = Binding.second.getAs<nonloc::LazyCompoundVal>()) {
      if (LCVal->getRegion() == Reg)
        return true;
    }
  }

  return false;
}

template <typename Condition, typename Process>
ProgramStateRef processIteratorPositions(ProgramStateRef State, Condition Cond,
                                         Process Proc) {
  auto &RegionMapFactory = State->get_context<IteratorRegionMap>();
  auto RegionMap = State->get<IteratorRegionMap>();
  bool Changed = false;
  for (const auto &Reg : RegionMap) {
    if (Cond(Reg.second)) {
      RegionMap = RegionMapFactory.add(RegionMap, Reg.first, Proc(Reg.second));
      Changed = true;
    }
  }

  if (Changed)
    State = State->set<IteratorRegionMap>(RegionMap);

  auto &SymbolMapFactory = State->get_context<IteratorSymbolMap>();
  auto SymbolMap = State->get<IteratorSymbolMap>();
  Changed = false;
  for (const auto &Sym : SymbolMap) {
    if (Cond(Sym.second)) {
      SymbolMap = SymbolMapFactory.add(SymbolMap, Sym.first, Proc(Sym.second));
      Changed = true;
    }
  }

  if (Changed)
    State = State->set<IteratorSymbolMap>(SymbolMap);

  return State;
}

ProgramStateRef invalidateAllIteratorPositions(ProgramStateRef State,
                                               const MemRegion *Cont) {
  auto MatchCont = [&](const IteratorPosition &Pos) {
    return Pos.getContainer() == Cont;
  };
  auto Invalidate = [&](const IteratorPosition &Pos) {
    return Pos.invalidate();
  };
  return processIteratorPositions(State, MatchCont, Invalidate);
}

ProgramStateRef
invalidateAllIteratorPositionsExcept(ProgramStateRef State,
                                     const MemRegion *Cont, SymbolRef Offset,
                                     BinaryOperator::Opcode Opc) {
  auto MatchContAndCompare = [&](const IteratorPosition &Pos) {
    return Pos.getContainer() == Cont &&
           !compare(State, Pos.getOffset(), Offset, Opc);
  };
  auto Invalidate = [&](const IteratorPosition &Pos) {
    return Pos.invalidate();
  };
  return processIteratorPositions(State, MatchContAndCompare, Invalidate);
}

ProgramStateRef invalidateIteratorPositions(ProgramStateRef State,
                                            SymbolRef Offset,
                                            BinaryOperator::Opcode Opc) {
  auto Compare = [&](const IteratorPosition &Pos) {
    return compare(State, Pos.getOffset(), Offset, Opc);
  };
  auto Invalidate = [&](const IteratorPosition &Pos) {
    return Pos.invalidate();
  };
  return processIteratorPositions(State, Compare, Invalidate);
}

ProgramStateRef invalidateIteratorPositions(ProgramStateRef State,
                                            SymbolRef Offset1,
                                            BinaryOperator::Opcode Opc1,
                                            SymbolRef Offset2,
                                            BinaryOperator::Opcode Opc2) {
  auto Compare = [&](const IteratorPosition &Pos) {
    return compare(State, Pos.getOffset(), Offset1, Opc1) &&
           compare(State, Pos.getOffset(), Offset2, Opc2);
  };
  auto Invalidate = [&](const IteratorPosition &Pos) {
    return Pos.invalidate();
  };
  return processIteratorPositions(State, Compare, Invalidate);
}

ProgramStateRef reassignAllIteratorPositions(ProgramStateRef State,
                                             const MemRegion *Cont,
                                             const MemRegion *NewCont) {
  auto MatchCont = [&](const IteratorPosition &Pos) {
    return Pos.getContainer() == Cont;
  };
  auto ReAssign = [&](const IteratorPosition &Pos) {
    return Pos.reAssign(NewCont);
  };
  return processIteratorPositions(State, MatchCont, ReAssign);
}

ProgramStateRef reassignAllIteratorPositionsUnless(ProgramStateRef State,
                                                   const MemRegion *Cont,
                                                   const MemRegion *NewCont,
                                                   SymbolRef Offset,
                                                   BinaryOperator::Opcode Opc) {
  auto MatchContAndCompare = [&](const IteratorPosition &Pos) {
    return Pos.getContainer() == Cont &&
    !compare(State, Pos.getOffset(), Offset, Opc);
  };
  auto ReAssign = [&](const IteratorPosition &Pos) {
    return Pos.reAssign(NewCont);
  };
  return processIteratorPositions(State, MatchContAndCompare, ReAssign);
}

// This function rebases symbolic expression `OldSym + Int` to `NewSym + Int`,
// `OldSym - Int` to `NewSym - Int` and  `OldSym` to `NewSym` in any iterator
// position offsets where `CondSym` is true.
ProgramStateRef rebaseSymbolInIteratorPositionsIf(
    ProgramStateRef State, SValBuilder &SVB, SymbolRef OldSym,
    SymbolRef NewSym, SymbolRef CondSym, BinaryOperator::Opcode Opc) {
  auto LessThanEnd = [&](const IteratorPosition &Pos) {
    return compare(State, Pos.getOffset(), CondSym, Opc);
  };
  auto RebaseSymbol = [&](const IteratorPosition &Pos) {
    return Pos.setTo(rebaseSymbol(State, SVB, Pos.getOffset(), OldSym,
                                   NewSym));
  };
  return processIteratorPositions(State, LessThanEnd, RebaseSymbol);
}

// This function rebases symbolic expression `OldExpr + Int` to `NewExpr + Int`,
// `OldExpr - Int` to `NewExpr - Int` and  `OldExpr` to `NewExpr` in expression
// `OrigExpr`.
SymbolRef rebaseSymbol(ProgramStateRef State, SValBuilder &SVB,
                       SymbolRef OrigExpr, SymbolRef OldExpr,
                       SymbolRef NewSym) {
  auto &SymMgr = SVB.getSymbolManager();
  auto Diff = SVB.evalBinOpNN(State, BO_Sub, nonloc::SymbolVal(OrigExpr),
                              nonloc::SymbolVal(OldExpr), 
                              SymMgr.getType(OrigExpr));

  const auto DiffInt = Diff.getAs<nonloc::ConcreteInt>();
  if (!DiffInt)
    return OrigExpr;

  return SVB.evalBinOpNN(State, BO_Add, *DiffInt, nonloc::SymbolVal(NewSym),
                         SymMgr.getType(OrigExpr)).getAsSymbol();
}

} // namespace

void ento::registerIteratorModeling(CheckerManager &mgr) {
  mgr.registerChecker<IteratorModeling>();
}

bool ento::shouldRegisterIteratorModeling(const LangOptions &LO) {
  return true;
}
