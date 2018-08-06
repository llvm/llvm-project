//===-- IteratorChecker.cpp ---------------------------------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {

// Abstract position of an iterator. This helps to handle all three kinds
// of operators in a common way by using a symbolic position.
struct IteratorPosition {
private:

  // Container the iterator belongs to
  const MemRegion *Cont;

  // Abstract offset
  const SymbolRef Offset;

  IteratorPosition(const MemRegion *C, SymbolRef Of)
      : Cont(C), Offset(Of) {}

public:
  const MemRegion *getContainer() const { return Cont; }
  SymbolRef getOffset() const { return Offset; }

  static IteratorPosition getPosition(const MemRegion *C, SymbolRef Of) {
    return IteratorPosition(C, Of);
  }

  IteratorPosition setTo(SymbolRef NewOf) const {
    return IteratorPosition(Cont, NewOf);
  }

  bool operator==(const IteratorPosition &X) const {
    return Cont == X.Cont && Offset == X.Offset;
  }

  bool operator!=(const IteratorPosition &X) const {
    return Cont != X.Cont || Offset != X.Offset;
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(Cont);
    ID.Add(Offset);
  }
};

typedef llvm::PointerUnion<const MemRegion *, SymbolRef> RegionOrSymbol;

// Structure to record the symbolic begin and end position of a container
struct ContainerData {
private:
  const SymbolRef Begin, End;

  ContainerData(SymbolRef B, SymbolRef E) : Begin(B), End(E) {}

public:
  static ContainerData fromBegin(SymbolRef B) {
    return ContainerData(B, nullptr);
  }

  static ContainerData fromEnd(SymbolRef E) {
    return ContainerData(nullptr, E);
  }

  SymbolRef getBegin() const { return Begin; }
  SymbolRef getEnd() const { return End; }

  ContainerData newBegin(SymbolRef B) const { return ContainerData(B, End); }

  ContainerData newEnd(SymbolRef E) const { return ContainerData(Begin, E); }

  bool operator==(const ContainerData &X) const {
    return Begin == X.Begin && End == X.End;
  }

  bool operator!=(const ContainerData &X) const {
    return Begin != X.Begin || End != X.End;
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.Add(Begin);
    ID.Add(End);
  }
};

// Structure fo recording iterator comparisons. We needed to retrieve the
// original comparison expression in assumptions.
struct IteratorComparison {
private:
  RegionOrSymbol Left, Right;
  bool Equality;

public:
  IteratorComparison(RegionOrSymbol L, RegionOrSymbol R, bool Eq)
      : Left(L), Right(R), Equality(Eq) {}

  RegionOrSymbol getLeft() const { return Left; }
  RegionOrSymbol getRight() const { return Right; }
  bool isEquality() const { return Equality; }
  bool operator==(const IteratorComparison &X) const {
    return Left == X.Left && Right == X.Right && Equality == X.Equality;
  }
  bool operator!=(const IteratorComparison &X) const {
    return Left != X.Left || Right != X.Right || Equality != X.Equality;
  }
  void Profile(llvm::FoldingSetNodeID &ID) const { ID.AddInteger(Equality); }
};

class IteratorChecker
    : public Checker<check::PreCall, check::PostCall,
                     check::PreStmt<CXXOperatorCallExpr>,
                     check::PostStmt<MaterializeTemporaryExpr>,
                     check::LiveSymbols, check::DeadSymbols,
                     eval::Assume> {

  std::unique_ptr<BugType> OutOfRangeBugType;

  void handleComparison(CheckerContext &C, const SVal &RetVal, const SVal &LVal,
                        const SVal &RVal, OverloadedOperatorKind Op) const;
  void verifyDereference(CheckerContext &C, const SVal &Val) const;
  void handleIncrement(CheckerContext &C, const SVal &RetVal, const SVal &Iter,
                       bool Postfix) const;
  void handleDecrement(CheckerContext &C, const SVal &RetVal, const SVal &Iter,
                       bool Postfix) const;
  void handleRandomIncrOrDecr(CheckerContext &C, OverloadedOperatorKind Op,
                              const SVal &RetVal, const SVal &LHS,
                              const SVal &RHS) const;
  void handleBegin(CheckerContext &C, const Expr *CE, const SVal &RetVal,
                   const SVal &Cont) const;
  void handleEnd(CheckerContext &C, const Expr *CE, const SVal &RetVal,
                 const SVal &Cont) const;
  void assignToContainer(CheckerContext &C, const Expr *CE, const SVal &RetVal,
                         const MemRegion *Cont) const;
  void verifyRandomIncrOrDecr(CheckerContext &C, OverloadedOperatorKind Op,
                              const SVal &RetVal, const SVal &LHS,
                              const SVal &RHS) const;
  void reportOutOfRangeBug(const StringRef &Message, const SVal &Val,
                           CheckerContext &C, ExplodedNode *ErrNode) const;

public:
  IteratorChecker();

  enum CheckKind {
    CK_IteratorRangeChecker,
    CK_NumCheckKinds
  };

  DefaultBool ChecksEnabled[CK_NumCheckKinds];
  CheckName CheckNames[CK_NumCheckKinds];

  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPreStmt(const CXXOperatorCallExpr *COCE, CheckerContext &C) const;
  void checkPostStmt(const MaterializeTemporaryExpr *MTE,
                     CheckerContext &C) const;
  void checkLiveSymbols(ProgramStateRef State, SymbolReaper &SR) const;
  void checkDeadSymbols(SymbolReaper &SR, CheckerContext &C) const;
  ProgramStateRef evalAssume(ProgramStateRef State, SVal Cond,
                             bool Assumption) const;
};
} // namespace

REGISTER_MAP_WITH_PROGRAMSTATE(IteratorSymbolMap, SymbolRef, IteratorPosition)
REGISTER_MAP_WITH_PROGRAMSTATE(IteratorRegionMap, const MemRegion *,
                               IteratorPosition)

REGISTER_MAP_WITH_PROGRAMSTATE(ContainerMap, const MemRegion *, ContainerData)

REGISTER_MAP_WITH_PROGRAMSTATE(IteratorComparisonMap, const SymExpr *,
                               IteratorComparison)

namespace {

bool isIteratorType(const QualType &Type);
bool isIterator(const CXXRecordDecl *CRD);
bool isBeginCall(const FunctionDecl *Func);
bool isEndCall(const FunctionDecl *Func);
bool isSimpleComparisonOperator(OverloadedOperatorKind OK);
bool isDereferenceOperator(OverloadedOperatorKind OK);
bool isIncrementOperator(OverloadedOperatorKind OK);
bool isDecrementOperator(OverloadedOperatorKind OK);
bool isRandomIncrOrDecrOperator(OverloadedOperatorKind OK);
BinaryOperator::Opcode getOpcode(const SymExpr *SE);
const RegionOrSymbol getRegionOrSymbol(const SVal &Val);
const ProgramStateRef processComparison(ProgramStateRef State,
                                        RegionOrSymbol LVal,
                                        RegionOrSymbol RVal, bool Equal);
const ProgramStateRef saveComparison(ProgramStateRef State,
                                     const SymExpr *Condition, const SVal &LVal,
                                     const SVal &RVal, bool Eq);
const IteratorComparison *loadComparison(ProgramStateRef State,
                                         const SymExpr *Condition);
SymbolRef getContainerBegin(ProgramStateRef State, const MemRegion *Cont);
SymbolRef getContainerEnd(ProgramStateRef State, const MemRegion *Cont);
ProgramStateRef createContainerBegin(ProgramStateRef State,
                                     const MemRegion *Cont,
                                     const SymbolRef Sym);
ProgramStateRef createContainerEnd(ProgramStateRef State, const MemRegion *Cont,
                                   const SymbolRef Sym);
const IteratorPosition *getIteratorPosition(ProgramStateRef State,
                                            const SVal &Val);
const IteratorPosition *getIteratorPosition(ProgramStateRef State,
                                            RegionOrSymbol RegOrSym);
ProgramStateRef setIteratorPosition(ProgramStateRef State, const SVal &Val,
                                    const IteratorPosition &Pos);
ProgramStateRef setIteratorPosition(ProgramStateRef State,
                                    RegionOrSymbol RegOrSym,
                                    const IteratorPosition &Pos);
ProgramStateRef removeIteratorPosition(ProgramStateRef State, const SVal &Val);
ProgramStateRef adjustIteratorPosition(ProgramStateRef State,
                                       RegionOrSymbol RegOrSym,
                                       const IteratorPosition &Pos, bool Equal);
ProgramStateRef relateIteratorPositions(ProgramStateRef State,
                                        const IteratorPosition &Pos1,
                                        const IteratorPosition &Pos2,
                                        bool Equal);
const ContainerData *getContainerData(ProgramStateRef State,
                                      const MemRegion *Cont);
ProgramStateRef setContainerData(ProgramStateRef State, const MemRegion *Cont,
                                 const ContainerData &CData);
bool isOutOfRange(ProgramStateRef State, const IteratorPosition &Pos);
bool isZero(ProgramStateRef State, const NonLoc &Val);
} // namespace

IteratorChecker::IteratorChecker() {
  OutOfRangeBugType.reset(
      new BugType(this, "Iterator out of range", "Misuse of STL APIs"));
  OutOfRangeBugType->setSuppressOnSink(true);
}

void IteratorChecker::checkPreCall(const CallEvent &Call,
                                   CheckerContext &C) const {
  // Check for out of range access
  const auto *Func = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!Func)
    return;

  if (Func->isOverloadedOperator()) {
    if (ChecksEnabled[CK_IteratorRangeChecker] &&
        isRandomIncrOrDecrOperator(Func->getOverloadedOperator())) {
      if (const auto *InstCall = dyn_cast<CXXInstanceCall>(&Call)) {
        // Check for out-of-range incrementions and decrementions
        if (Call.getNumArgs() >= 1) {
          verifyRandomIncrOrDecr(C, Func->getOverloadedOperator(),
                                 Call.getReturnValue(),
                                 InstCall->getCXXThisVal(), Call.getArgSVal(0));
        }
      } else {
        if (Call.getNumArgs() >= 2) {
          verifyRandomIncrOrDecr(C, Func->getOverloadedOperator(),
                                 Call.getReturnValue(), Call.getArgSVal(0),
                                 Call.getArgSVal(1));
        }
      }
    } else if (ChecksEnabled[CK_IteratorRangeChecker] &&
               isDereferenceOperator(Func->getOverloadedOperator())) {
      // Check for dereference of out-of-range iterators
      if (const auto *InstCall = dyn_cast<CXXInstanceCall>(&Call)) {
        verifyDereference(C, InstCall->getCXXThisVal());
      } else {
        verifyDereference(C, Call.getArgSVal(0));
      }
    }
  }
}

void IteratorChecker::checkPostCall(const CallEvent &Call,
                                    CheckerContext &C) const {
  // Record new iterator positions and iterator position changes
  const auto *Func = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!Func)
    return;

  if (Func->isOverloadedOperator()) {
    const auto Op = Func->getOverloadedOperator();
    if (isSimpleComparisonOperator(Op)) {
      if (const auto *InstCall = dyn_cast<CXXInstanceCall>(&Call)) {
        handleComparison(C, Call.getReturnValue(), InstCall->getCXXThisVal(),
                         Call.getArgSVal(0), Op);
      } else {
        handleComparison(C, Call.getReturnValue(), Call.getArgSVal(0),
                         Call.getArgSVal(1), Op);
      }
    } else if (isRandomIncrOrDecrOperator(Func->getOverloadedOperator())) {
      if (const auto *InstCall = dyn_cast<CXXInstanceCall>(&Call)) {
        if (Call.getNumArgs() >= 1) {
          handleRandomIncrOrDecr(C, Func->getOverloadedOperator(),
                                 Call.getReturnValue(),
                                 InstCall->getCXXThisVal(), Call.getArgSVal(0));
        }
      } else {
        if (Call.getNumArgs() >= 2) {
          handleRandomIncrOrDecr(C, Func->getOverloadedOperator(),
                                 Call.getReturnValue(), Call.getArgSVal(0),
                                 Call.getArgSVal(1));
        }
      }
    } else if (isIncrementOperator(Func->getOverloadedOperator())) {
      if (const auto *InstCall = dyn_cast<CXXInstanceCall>(&Call)) {
        handleIncrement(C, Call.getReturnValue(), InstCall->getCXXThisVal(),
                        Call.getNumArgs());
      } else {
        handleIncrement(C, Call.getReturnValue(), Call.getArgSVal(0),
                        Call.getNumArgs());
      }
    } else if (isDecrementOperator(Func->getOverloadedOperator())) {
      if (const auto *InstCall = dyn_cast<CXXInstanceCall>(&Call)) {
        handleDecrement(C, Call.getReturnValue(), InstCall->getCXXThisVal(),
                        Call.getNumArgs());
      } else {
        handleDecrement(C, Call.getReturnValue(), Call.getArgSVal(0),
                        Call.getNumArgs());
      }
    }
  } else {
    const auto *OrigExpr = Call.getOriginExpr();
    if (!OrigExpr)
      return;

    if (!isIteratorType(Call.getResultType()))
      return;

    auto State = C.getState();
    // Already bound to container?
    if (getIteratorPosition(State, Call.getReturnValue()))
      return;

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

void IteratorChecker::checkPreStmt(const CXXOperatorCallExpr *COCE,
                                   CheckerContext &C) const {
  const auto *ThisExpr = COCE->getArg(0);

  auto State = C.getState();
  const auto *LCtx = C.getLocationContext();

  const auto CurrentThis = State->getSVal(ThisExpr, LCtx);
  if (const auto *Reg = CurrentThis.getAsRegion()) {
    if (!Reg->getAs<CXXTempObjectRegion>())
      return;
    const auto OldState = C.getPredecessor()->getFirstPred()->getState();
    const auto OldThis = OldState->getSVal(ThisExpr, LCtx);
    // FIXME: This solution is unreliable. It may happen that another checker
    //        subscribes to the pre-statement check of `CXXOperatorCallExpr`
    //        and adds a transition before us. The proper fix is to make the
    //        CFG provide a `ConstructionContext` for the `CXXOperatorCallExpr`,
    //        which would turn the corresponding `CFGStmt` element into a
    //        `CFGCXXRecordTypedCall` element, which will allow `ExprEngine` to
    //        foresee that the `begin()`/`end()` call constructs the object
    //        directly in the temporary region that `CXXOperatorCallExpr` takes
    //        as its implicit object argument.
    const auto *Pos = getIteratorPosition(OldState, OldThis);
    if (!Pos)
      return;
    State = setIteratorPosition(State, CurrentThis, *Pos);
    C.addTransition(State);
  }
}

void IteratorChecker::checkPostStmt(const MaterializeTemporaryExpr *MTE,
                                    CheckerContext &C) const {
  /* Transfer iterator state to temporary objects */
  auto State = C.getState();
  const auto *Pos =
      getIteratorPosition(State, C.getSVal(MTE->GetTemporaryExpr()));
  if (!Pos)
    return;
  State = setIteratorPosition(State, C.getSVal(MTE), *Pos);
  C.addTransition(State);
}

void IteratorChecker::checkLiveSymbols(ProgramStateRef State,
                                       SymbolReaper &SR) const {
  // Keep symbolic expressions of iterator positions, container begins and ends
  // alive
  auto RegionMap = State->get<IteratorRegionMap>();
  for (const auto Reg : RegionMap) {
    const auto Offset = Reg.second.getOffset();
    for (auto i = Offset->symbol_begin(); i != Offset->symbol_end(); ++i)
      if (isa<SymbolData>(*i))
        SR.markLive(*i);
  }

  auto SymbolMap = State->get<IteratorSymbolMap>();
  for (const auto Sym : SymbolMap) {
    const auto Offset = Sym.second.getOffset();
    for (auto i = Offset->symbol_begin(); i != Offset->symbol_end(); ++i)
      if (isa<SymbolData>(*i))
        SR.markLive(*i);
  }

  auto ContMap = State->get<ContainerMap>();
  for (const auto Cont : ContMap) {
    const auto CData = Cont.second;
    if (CData.getBegin()) {
      SR.markLive(CData.getBegin());
    }
    if (CData.getEnd()) {
      SR.markLive(CData.getEnd());
    }
  }
}

void IteratorChecker::checkDeadSymbols(SymbolReaper &SR,
                                       CheckerContext &C) const {
  // Cleanup
  auto State = C.getState();

  auto RegionMap = State->get<IteratorRegionMap>();
  for (const auto Reg : RegionMap) {
    if (!SR.isLiveRegion(Reg.first)) {
      State = State->remove<IteratorRegionMap>(Reg.first);
    }
  }

  auto SymbolMap = State->get<IteratorSymbolMap>();
  for (const auto Sym : SymbolMap) {
    if (!SR.isLive(Sym.first)) {
      State = State->remove<IteratorSymbolMap>(Sym.first);
    }
  }

  auto ContMap = State->get<ContainerMap>();
  for (const auto Cont : ContMap) {
    if (!SR.isLiveRegion(Cont.first)) {
      State = State->remove<ContainerMap>(Cont.first);
    }
  }

  auto ComparisonMap = State->get<IteratorComparisonMap>();
  for (const auto Comp : ComparisonMap) {
    if (!SR.isLive(Comp.first)) {
      State = State->remove<IteratorComparisonMap>(Comp.first);
    }
  }
}

ProgramStateRef IteratorChecker::evalAssume(ProgramStateRef State, SVal Cond,
                                            bool Assumption) const {
  // Load recorded comparison and transfer iterator state between sides
  // according to comparison operator and assumption
  const auto *SE = Cond.getAsSymExpr();
  if (!SE)
    return State;

  auto Opc = getOpcode(SE);
  if (Opc != BO_EQ && Opc != BO_NE)
    return State;

  bool Negated = false;
  const auto *Comp = loadComparison(State, SE);
  if (!Comp) {
    // Try negated comparison, which is a SymExpr to 0 integer comparison
    const auto *SIE = dyn_cast<SymIntExpr>(SE);
    if (!SIE)
      return State;

    if (SIE->getRHS() != 0)
      return State;

    SE = SIE->getLHS();
    Negated = SIE->getOpcode() == BO_EQ; // Equal to zero means negation
    Opc = getOpcode(SE);
    if (Opc != BO_EQ && Opc != BO_NE)
      return State;

    Comp = loadComparison(State, SE);
    if (!Comp)
      return State;
  }

  return processComparison(State, Comp->getLeft(), Comp->getRight(),
                           (Comp->isEquality() == Assumption) != Negated);
}

void IteratorChecker::handleComparison(CheckerContext &C, const SVal &RetVal,
                                       const SVal &LVal, const SVal &RVal,
                                       OverloadedOperatorKind Op) const {
  // Record the operands and the operator of the comparison for the next
  // evalAssume, if the result is a symbolic expression. If it is a concrete
  // value (only one branch is possible), then transfer the state between
  // the operands according to the operator and the result
  auto State = C.getState();
  if (const auto *Condition = RetVal.getAsSymbolicExpression()) {
    const auto *LPos = getIteratorPosition(State, LVal);
    const auto *RPos = getIteratorPosition(State, RVal);
    if (!LPos && !RPos)
      return;
    State = saveComparison(State, Condition, LVal, RVal, Op == OO_EqualEqual);
    C.addTransition(State);
  } else if (const auto TruthVal = RetVal.getAs<nonloc::ConcreteInt>()) {
    if ((State = processComparison(
             State, getRegionOrSymbol(LVal), getRegionOrSymbol(RVal),
             (Op == OO_EqualEqual) == (TruthVal->getValue() != 0)))) {
      C.addTransition(State);
    } else {
      C.generateSink(State, C.getPredecessor());
    }
  }
}

void IteratorChecker::verifyDereference(CheckerContext &C,
                                        const SVal &Val) const {
  auto State = C.getState();
  const auto *Pos = getIteratorPosition(State, Val);
  if (Pos && isOutOfRange(State, *Pos)) {
    // If I do not put a tag here, some range tests will fail
    static CheckerProgramPointTag Tag("IteratorRangeChecker",
                                      "IteratorOutOfRange");
    auto *N = C.generateNonFatalErrorNode(State, &Tag);
    if (!N)
      return;
    reportOutOfRangeBug("Iterator accessed outside of its range.", Val, C, N);
  }
}

void IteratorChecker::handleIncrement(CheckerContext &C, const SVal &RetVal,
                                      const SVal &Iter, bool Postfix) const {
  // Increment the symbolic expressions which represents the position of the
  // iterator
  auto State = C.getState();
  const auto *Pos = getIteratorPosition(State, Iter);
  if (Pos) {
    auto &SymMgr = C.getSymbolManager();
    auto &BVF = SymMgr.getBasicVals();
    auto &SVB = C.getSValBuilder();
    const auto OldOffset = Pos->getOffset();
    auto NewOffset =
      SVB.evalBinOp(State, BO_Add,
                    nonloc::SymbolVal(OldOffset),
                    nonloc::ConcreteInt(BVF.getValue(llvm::APSInt::get(1))),
                    SymMgr.getType(OldOffset)).getAsSymbol();
    auto NewPos = Pos->setTo(NewOffset);
    State = setIteratorPosition(State, Iter, NewPos);
    State = setIteratorPosition(State, RetVal, Postfix ? *Pos : NewPos);
    C.addTransition(State);
  }
}

void IteratorChecker::handleDecrement(CheckerContext &C, const SVal &RetVal,
                                      const SVal &Iter, bool Postfix) const {
  // Decrement the symbolic expressions which represents the position of the
  // iterator
  auto State = C.getState();
  const auto *Pos = getIteratorPosition(State, Iter);
  if (Pos) {
    auto &SymMgr = C.getSymbolManager();
    auto &BVF = SymMgr.getBasicVals();
    auto &SVB = C.getSValBuilder();
    const auto OldOffset = Pos->getOffset();
    auto NewOffset =
      SVB.evalBinOp(State, BO_Sub,
                    nonloc::SymbolVal(OldOffset),
                    nonloc::ConcreteInt(BVF.getValue(llvm::APSInt::get(1))),
                    SymMgr.getType(OldOffset)).getAsSymbol();
    auto NewPos = Pos->setTo(NewOffset);
    State = setIteratorPosition(State, Iter, NewPos);
    State = setIteratorPosition(State, RetVal, Postfix ? *Pos : NewPos);
    C.addTransition(State);
  }
}

// This function tells the analyzer's engine that symbols produced by our
// checker, most notably iterator positions, are relatively small.
// A distance between items in the container should not be very large.
// By assuming that it is within around 1/8 of the address space,
// we can help the analyzer perform operations on these symbols
// without being afraid of integer overflows.
// FIXME: Should we provide it as an API, so that all checkers could use it?
static ProgramStateRef assumeNoOverflow(ProgramStateRef State, SymbolRef Sym,
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

void IteratorChecker::handleRandomIncrOrDecr(CheckerContext &C,
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

  auto &SymMgr = C.getSymbolManager();
  auto &SVB = C.getSValBuilder();
  auto BinOp = (Op == OO_Plus || Op == OO_PlusEqual) ? BO_Add : BO_Sub;
  const auto OldOffset = Pos->getOffset();
  SymbolRef NewOffset;
  if (const auto intValue = value->getAs<nonloc::ConcreteInt>()) {
    // For concrete integers we can calculate the new position
    NewOffset = SVB.evalBinOp(State, BinOp, nonloc::SymbolVal(OldOffset),
                              *intValue,
                              SymMgr.getType(OldOffset)).getAsSymbol();
  } else {
    // For other symbols create a new symbol to keep expressions simple
    const auto &LCtx = C.getLocationContext();
    NewOffset = SymMgr.conjureSymbol(nullptr, LCtx, SymMgr.getType(OldOffset),
                                     C.blockCount());
    State = assumeNoOverflow(State, NewOffset, 4);
  }
  auto NewPos = Pos->setTo(NewOffset);
  auto &TgtVal = (Op == OO_PlusEqual || Op == OO_MinusEqual) ? LHS : RetVal;
  State = setIteratorPosition(State, TgtVal, NewPos);
  C.addTransition(State);
}

void IteratorChecker::verifyRandomIncrOrDecr(CheckerContext &C,
                                             OverloadedOperatorKind Op,
                                             const SVal &RetVal,
                                             const SVal &LHS,
                                             const SVal &RHS) const {
  auto State = C.getState();

  // If the iterator is initially inside its range, then the operation is valid
  const auto *Pos = getIteratorPosition(State, LHS);
  if (!Pos || !isOutOfRange(State, *Pos))
    return;

  auto value = RHS;
  if (auto loc = RHS.getAs<Loc>()) {
    value = State->getRawSVal(*loc);
  }

  // Incremention or decremention by 0 is never bug
  if (isZero(State, value.castAs<NonLoc>()))
    return;

  auto &SymMgr = C.getSymbolManager();
  auto &SVB = C.getSValBuilder();
  auto BinOp = (Op == OO_Plus || Op == OO_PlusEqual) ? BO_Add : BO_Sub;
  const auto OldOffset = Pos->getOffset();
  const auto intValue = value.getAs<nonloc::ConcreteInt>();
  if (!intValue)
    return;

  auto NewOffset = SVB.evalBinOp(State, BinOp, nonloc::SymbolVal(OldOffset),
                                 *intValue,
                                 SymMgr.getType(OldOffset)).getAsSymbol();
  auto NewPos = Pos->setTo(NewOffset);

  // If out of range, the only valid operation is to step into the range
  if (isOutOfRange(State, NewPos)) {
    auto *N = C.generateNonFatalErrorNode(State);
    if (!N)
      return;
    reportOutOfRangeBug("Iterator accessed past its end.", LHS, C, N);
  }
}

void IteratorChecker::handleBegin(CheckerContext &C, const Expr *CE,
                                  const SVal &RetVal, const SVal &Cont) const {
  const auto *ContReg = Cont.getAsRegion();
  if (!ContReg)
    return;

  while (const auto *CBOR = ContReg->getAs<CXXBaseObjectRegion>()) {
    ContReg = CBOR->getSuperRegion();
  }

  // If the container already has a begin symbol then use it. Otherwise first
  // create a new one.
  auto State = C.getState();
  auto BeginSym = getContainerBegin(State, ContReg);
  if (!BeginSym) {
    auto &SymMgr = C.getSymbolManager();
    BeginSym = SymMgr.conjureSymbol(CE, C.getLocationContext(),
                                    C.getASTContext().LongTy, C.blockCount());
    State = assumeNoOverflow(State, BeginSym, 4);
    State = createContainerBegin(State, ContReg, BeginSym);
  }
  State = setIteratorPosition(State, RetVal,
                              IteratorPosition::getPosition(ContReg, BeginSym));
  C.addTransition(State);
}

void IteratorChecker::handleEnd(CheckerContext &C, const Expr *CE,
                                const SVal &RetVal, const SVal &Cont) const {
  const auto *ContReg = Cont.getAsRegion();
  if (!ContReg)
    return;

  while (const auto *CBOR = ContReg->getAs<CXXBaseObjectRegion>()) {
    ContReg = CBOR->getSuperRegion();
  }

  // If the container already has an end symbol then use it. Otherwise first
  // create a new one.
  auto State = C.getState();
  auto EndSym = getContainerEnd(State, ContReg);
  if (!EndSym) {
    auto &SymMgr = C.getSymbolManager();
    EndSym = SymMgr.conjureSymbol(CE, C.getLocationContext(),
                                  C.getASTContext().LongTy, C.blockCount());
    State = assumeNoOverflow(State, EndSym, 4);
    State = createContainerEnd(State, ContReg, EndSym);
  }
  State = setIteratorPosition(State, RetVal,
                              IteratorPosition::getPosition(ContReg, EndSym));
  C.addTransition(State);
}

void IteratorChecker::assignToContainer(CheckerContext &C, const Expr *CE,
                                        const SVal &RetVal,
                                        const MemRegion *Cont) const {
  while (const auto *CBOR = Cont->getAs<CXXBaseObjectRegion>()) {
    Cont = CBOR->getSuperRegion();
  }

  auto State = C.getState();
  auto &SymMgr = C.getSymbolManager();
  auto Sym = SymMgr.conjureSymbol(CE, C.getLocationContext(),
                                  C.getASTContext().LongTy, C.blockCount());
  State = assumeNoOverflow(State, Sym, 4);
  State = setIteratorPosition(State, RetVal,
                              IteratorPosition::getPosition(Cont, Sym));
  C.addTransition(State);
}

void IteratorChecker::reportOutOfRangeBug(const StringRef &Message,
                                          const SVal &Val, CheckerContext &C,
                                          ExplodedNode *ErrNode) const {
  auto R = llvm::make_unique<BugReport>(*OutOfRangeBugType, Message, ErrNode);
  R->markInteresting(Val);
  C.emitReport(std::move(R));
}

namespace {

bool isLess(ProgramStateRef State, SymbolRef Sym1, SymbolRef Sym2);
bool isGreaterOrEqual(ProgramStateRef State, SymbolRef Sym1, SymbolRef Sym2);
bool compare(ProgramStateRef State, SymbolRef Sym1, SymbolRef Sym2,
             BinaryOperator::Opcode Opc);
bool compare(ProgramStateRef State, NonLoc NL1, NonLoc NL2,
             BinaryOperator::Opcode Opc);

bool isIteratorType(const QualType &Type) {
  if (Type->isPointerType())
    return true;

  const auto *CRD = Type->getUnqualifiedDesugaredType()->getAsCXXRecordDecl();
  return isIterator(CRD);
}

bool isIterator(const CXXRecordDecl *CRD) {
  if (!CRD)
    return false;

  const auto Name = CRD->getName();
  if (!(Name.endswith_lower("iterator") || Name.endswith_lower("iter") ||
        Name.endswith_lower("it")))
    return false;

  bool HasCopyCtor = false, HasCopyAssign = true, HasDtor = false,
       HasPreIncrOp = false, HasPostIncrOp = false, HasDerefOp = false;
  for (const auto *Method : CRD->methods()) {
    if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(Method)) {
      if (Ctor->isCopyConstructor()) {
        HasCopyCtor = !Ctor->isDeleted() && Ctor->getAccess() == AS_public;
      }
      continue;
    }
    if (const auto *Dtor = dyn_cast<CXXDestructorDecl>(Method)) {
      HasDtor = !Dtor->isDeleted() && Dtor->getAccess() == AS_public;
      continue;
    }
    if (Method->isCopyAssignmentOperator()) {
      HasCopyAssign = !Method->isDeleted() && Method->getAccess() == AS_public;
      continue;
    }
    if (!Method->isOverloadedOperator())
      continue;
    const auto OPK = Method->getOverloadedOperator();
    if (OPK == OO_PlusPlus) {
      HasPreIncrOp = HasPreIncrOp || (Method->getNumParams() == 0);
      HasPostIncrOp = HasPostIncrOp || (Method->getNumParams() == 1);
      continue;
    }
    if (OPK == OO_Star) {
      HasDerefOp = (Method->getNumParams() == 0);
      continue;
    }
  }

  return HasCopyCtor && HasCopyAssign && HasDtor && HasPreIncrOp &&
         HasPostIncrOp && HasDerefOp;
}

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

bool isSimpleComparisonOperator(OverloadedOperatorKind OK) {
  return OK == OO_EqualEqual || OK == OO_ExclaimEqual;
}

bool isDereferenceOperator(OverloadedOperatorKind OK) {
  return OK == OO_Star || OK == OO_Arrow || OK == OO_ArrowStar ||
         OK == OO_Subscript;
}

bool isIncrementOperator(OverloadedOperatorKind OK) {
  return OK == OO_PlusPlus;
}

bool isDecrementOperator(OverloadedOperatorKind OK) {
  return OK == OO_MinusMinus;
}

bool isRandomIncrOrDecrOperator(OverloadedOperatorKind OK) {
  return OK == OO_Plus || OK == OO_PlusEqual || OK == OO_Minus ||
         OK == OO_MinusEqual;
}

BinaryOperator::Opcode getOpcode(const SymExpr *SE) {
  if (const auto *BSE = dyn_cast<BinarySymExpr>(SE)) {
    return BSE->getOpcode();
  } else if (const auto *SC = dyn_cast<SymbolConjured>(SE)) {
    const auto *COE = dyn_cast_or_null<CXXOperatorCallExpr>(SC->getStmt());
    if (!COE)
      return BO_Comma; // Extremal value, neither EQ nor NE
    if (COE->getOperator() == OO_EqualEqual) {
      return BO_EQ;
    } else if (COE->getOperator() == OO_ExclaimEqual) {
      return BO_NE;
    }
    return BO_Comma; // Extremal value, neither EQ nor NE
  }
  return BO_Comma; // Extremal value, neither EQ nor NE
}

const RegionOrSymbol getRegionOrSymbol(const SVal &Val) {
  if (const auto Reg = Val.getAsRegion()) {
    return Reg;
  } else if (const auto Sym = Val.getAsSymbol()) {
    return Sym;
  } else if (const auto LCVal = Val.getAs<nonloc::LazyCompoundVal>()) {
    return LCVal->getRegion();
  }
  return RegionOrSymbol();
}

const ProgramStateRef processComparison(ProgramStateRef State,
                                        RegionOrSymbol LVal,
                                        RegionOrSymbol RVal, bool Equal) {
  const auto *LPos = getIteratorPosition(State, LVal);
  const auto *RPos = getIteratorPosition(State, RVal);
  if (LPos && !RPos) {
    State = adjustIteratorPosition(State, RVal, *LPos, Equal);
  } else if (!LPos && RPos) {
    State = adjustIteratorPosition(State, LVal, *RPos, Equal);
  } else if (LPos && RPos) {
    State = relateIteratorPositions(State, *LPos, *RPos, Equal);
  }
  return State;
}

const ProgramStateRef saveComparison(ProgramStateRef State,
                                     const SymExpr *Condition, const SVal &LVal,
                                     const SVal &RVal, bool Eq) {
  const auto Left = getRegionOrSymbol(LVal);
  const auto Right = getRegionOrSymbol(RVal);
  if (!Left || !Right)
    return State;
  return State->set<IteratorComparisonMap>(Condition,
                                           IteratorComparison(Left, Right, Eq));
}

const IteratorComparison *loadComparison(ProgramStateRef State,
                                         const SymExpr *Condition) {
  return State->get<IteratorComparisonMap>(Condition);
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
                                     const MemRegion *Cont,
                                     const SymbolRef Sym) {
  // Only create if it does not exist
  const auto *CDataPtr = getContainerData(State, Cont);
  if (CDataPtr) {
    if (CDataPtr->getBegin()) {
      return State;
    }
    const auto CData = CDataPtr->newBegin(Sym);
    return setContainerData(State, Cont, CData);
  }
  const auto CData = ContainerData::fromBegin(Sym);
  return setContainerData(State, Cont, CData);
}

ProgramStateRef createContainerEnd(ProgramStateRef State, const MemRegion *Cont,
                                   const SymbolRef Sym) {
  // Only create if it does not exist
  const auto *CDataPtr = getContainerData(State, Cont);
  if (CDataPtr) {
    if (CDataPtr->getEnd()) {
      return State;
    }
    const auto CData = CDataPtr->newEnd(Sym);
    return setContainerData(State, Cont, CData);
  }
  const auto CData = ContainerData::fromEnd(Sym);
  return setContainerData(State, Cont, CData);
}

const ContainerData *getContainerData(ProgramStateRef State,
                                      const MemRegion *Cont) {
  return State->get<ContainerMap>(Cont);
}

ProgramStateRef setContainerData(ProgramStateRef State, const MemRegion *Cont,
                                 const ContainerData &CData) {
  return State->set<ContainerMap>(Cont, CData);
}

const IteratorPosition *getIteratorPosition(ProgramStateRef State,
                                            const SVal &Val) {
  if (const auto Reg = Val.getAsRegion()) {
    return State->get<IteratorRegionMap>(Reg);
  } else if (const auto Sym = Val.getAsSymbol()) {
    return State->get<IteratorSymbolMap>(Sym);
  } else if (const auto LCVal = Val.getAs<nonloc::LazyCompoundVal>()) {
    return State->get<IteratorRegionMap>(LCVal->getRegion());
  }
  return nullptr;
}

const IteratorPosition *getIteratorPosition(ProgramStateRef State,
                                            RegionOrSymbol RegOrSym) {
  if (RegOrSym.is<const MemRegion *>()) {
    return State->get<IteratorRegionMap>(RegOrSym.get<const MemRegion *>());
  } else if (RegOrSym.is<SymbolRef>()) {
    return State->get<IteratorSymbolMap>(RegOrSym.get<SymbolRef>());
  }
  return nullptr;
}

ProgramStateRef setIteratorPosition(ProgramStateRef State, const SVal &Val,
                                    const IteratorPosition &Pos) {
  if (const auto Reg = Val.getAsRegion()) {
    return State->set<IteratorRegionMap>(Reg, Pos);
  } else if (const auto Sym = Val.getAsSymbol()) {
    return State->set<IteratorSymbolMap>(Sym, Pos);
  } else if (const auto LCVal = Val.getAs<nonloc::LazyCompoundVal>()) {
    return State->set<IteratorRegionMap>(LCVal->getRegion(), Pos);
  }
  return nullptr;
}

ProgramStateRef setIteratorPosition(ProgramStateRef State,
                                    RegionOrSymbol RegOrSym,
                                    const IteratorPosition &Pos) {
  if (RegOrSym.is<const MemRegion *>()) {
    return State->set<IteratorRegionMap>(RegOrSym.get<const MemRegion *>(),
                                         Pos);
  } else if (RegOrSym.is<SymbolRef>()) {
    return State->set<IteratorSymbolMap>(RegOrSym.get<SymbolRef>(), Pos);
  }
  return nullptr;
}

ProgramStateRef removeIteratorPosition(ProgramStateRef State, const SVal &Val) {
  if (const auto Reg = Val.getAsRegion()) {
    return State->remove<IteratorRegionMap>(Reg);
  } else if (const auto Sym = Val.getAsSymbol()) {
    return State->remove<IteratorSymbolMap>(Sym);
  } else if (const auto LCVal = Val.getAs<nonloc::LazyCompoundVal>()) {
    return State->remove<IteratorRegionMap>(LCVal->getRegion());
  }
  return nullptr;
}

ProgramStateRef adjustIteratorPosition(ProgramStateRef State,
                                       RegionOrSymbol RegOrSym,
                                       const IteratorPosition &Pos,
                                       bool Equal) {
  if (Equal) {
    return setIteratorPosition(State, RegOrSym, Pos);
  } else {
    return State;
  }
}

ProgramStateRef relateIteratorPositions(ProgramStateRef State,
                                        const IteratorPosition &Pos1,
                                        const IteratorPosition &Pos2,
                                        bool Equal) {
  auto &SVB = State->getStateManager().getSValBuilder();

  // FIXME: This code should be reworked as follows:
  // 1. Subtract the operands using evalBinOp().
  // 2. Assume that the result doesn't overflow.
  // 3. Compare the result to 0.
  // 4. Assume the result of the comparison.
  const auto comparison =
      SVB.evalBinOp(State, BO_EQ, nonloc::SymbolVal(Pos1.getOffset()),
                    nonloc::SymbolVal(Pos2.getOffset()),
                    SVB.getConditionType());

  assert(comparison.getAs<DefinedSVal>() &&
    "Symbol comparison must be a `DefinedSVal`");

  auto NewState = State->assume(comparison.castAs<DefinedSVal>(), Equal);
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

bool isZero(ProgramStateRef State, const NonLoc &Val) {
  auto &BVF = State->getBasicVals();
  return compare(State, Val,
                 nonloc::ConcreteInt(BVF.getValue(llvm::APSInt::get(0))),
                 BO_EQ);
}

bool isOutOfRange(ProgramStateRef State, const IteratorPosition &Pos) {
  const auto *Cont = Pos.getContainer();
  const auto *CData = getContainerData(State, Cont);
  if (!CData)
    return false;

  // Out of range means less than the begin symbol or greater or equal to the
  // end symbol.

  const auto Beg = CData->getBegin();
  if (Beg) {
    if (isLess(State, Pos.getOffset(), Beg)) {
      return true;
    }
  }

  const auto End = CData->getEnd();
  if (End) {
    if (isGreaterOrEqual(State, Pos.getOffset(), End)) {
      return true;
    }
  }

  return false;
}

bool isLess(ProgramStateRef State, SymbolRef Sym1, SymbolRef Sym2) {
  return compare(State, Sym1, Sym2, BO_LT);
}

bool isGreaterOrEqual(ProgramStateRef State, SymbolRef Sym1, SymbolRef Sym2) {
  return compare(State, Sym1, Sym2, BO_GE);
}

bool compare(ProgramStateRef State, SymbolRef Sym1, SymbolRef Sym2,
             BinaryOperator::Opcode Opc) {
  return compare(State, nonloc::SymbolVal(Sym1), nonloc::SymbolVal(Sym2), Opc);
}

bool compare(ProgramStateRef State, NonLoc NL1, NonLoc NL2,
             BinaryOperator::Opcode Opc) {
  auto &SVB = State->getStateManager().getSValBuilder();

  const auto comparison =
    SVB.evalBinOp(State, Opc, NL1, NL2, SVB.getConditionType());

  assert(comparison.getAs<DefinedSVal>() &&
    "Symbol comparison must be a `DefinedSVal`");

  return !State->assume(comparison.castAs<DefinedSVal>(), false);
}

} // namespace

#define REGISTER_CHECKER(name)                                                 \
  void ento::register##name(CheckerManager &Mgr) {                             \
    auto *checker = Mgr.registerChecker<IteratorChecker>();                    \
    checker->ChecksEnabled[IteratorChecker::CK_##name] = true;                 \
    checker->CheckNames[IteratorChecker::CK_##name] =                          \
        Mgr.getCurrentCheckName();                                             \
  }

REGISTER_CHECKER(IteratorRangeChecker)

