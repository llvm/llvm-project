//===-- IteratorModeling.cpp --------------------------------------*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a modeling-checker for modeling STL iterator-like iterators.
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
  void assignToContainer(CheckerContext &C, const Expr *CE, const SVal &RetVal,
                         const MemRegion *Cont) const;
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

bool isSimpleComparisonOperator(OverloadedOperatorKind OK);
ProgramStateRef removeIteratorPosition(ProgramStateRef State, const SVal &Val);
ProgramStateRef relateSymbols(ProgramStateRef State, SymbolRef Sym1,
                              SymbolRef Sym2, bool Equal);
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
    if (isSimpleComparisonOperator(Op)) {
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
    if (!isIteratorType(Call.getResultType()))
      return;

    const auto *OrigExpr = Call.getOriginExpr();
    if (!OrigExpr)
      return;

    auto State = C.getState();

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
  // Keep symbolic expressions of iterator positions alive
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

void IteratorModeling::assignToContainer(CheckerContext &C, const Expr *CE,
                                         const SVal &RetVal,
                                         const MemRegion *Cont) const {
  Cont = Cont->getMostDerivedObjectRegion();

  auto State = C.getState();
  const auto *LCtx = C.getLocationContext();
  State = createIteratorPosition(State, RetVal, Cont, CE, LCtx, C.blockCount());

  C.addTransition(State);
}

void IteratorModeling::printState(raw_ostream &Out, ProgramStateRef State,
                                  const char *NL, const char *Sep) const {
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

bool isSimpleComparisonOperator(OverloadedOperatorKind OK) {
  return OK == OO_EqualEqual || OK == OO_ExclaimEqual;
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

} // namespace

void ento::registerIteratorModeling(CheckerManager &mgr) {
  mgr.registerChecker<IteratorModeling>();
}

bool ento::shouldRegisterIteratorModeling(const LangOptions &LO) {
  return true;
}
