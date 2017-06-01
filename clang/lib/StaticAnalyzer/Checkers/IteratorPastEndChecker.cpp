//===-- IteratorPastEndChecker.cpp --------------------------------*- C++ -*--//
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

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

#include <utility>

using namespace clang;
using namespace ento;

namespace {
struct IteratorPosition {
private:
  enum Kind { InRange, OutofRange } K;
  IteratorPosition(Kind InK) : K(InK) {}

public:
  bool isInRange() const { return K == InRange; }
  bool isOutofRange() const { return K == OutofRange; }

  static IteratorPosition getInRange() { return IteratorPosition(InRange); }
  static IteratorPosition getOutofRange() {
    return IteratorPosition(OutofRange);
  }

  bool operator==(const IteratorPosition &X) const { return K == X.K; }
  bool operator!=(const IteratorPosition &X) const { return K != X.K; }
  void Profile(llvm::FoldingSetNodeID &ID) const { ID.AddInteger(K); }
};

typedef llvm::PointerUnion<const MemRegion *, SymbolRef> RegionOrSymbol;

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

class IteratorPastEndChecker
    : public Checker<
          check::PreCall, check::PostCall, check::PreStmt<CXXOperatorCallExpr>,
          check::PostStmt<CXXConstructExpr>, check::PostStmt<DeclStmt>,
          check::PostStmt<MaterializeTemporaryExpr>, check::BeginFunction,
          check::DeadSymbols, eval::Assume, eval::Call> {
  mutable IdentifierInfo *II_find = nullptr,
                         *II_find_end = nullptr, *II_find_first_of = nullptr,
                         *II_find_if = nullptr, *II_find_if_not = nullptr,
                         *II_lower_bound = nullptr, *II_upper_bound = nullptr,
                         *II_search = nullptr, *II_search_n = nullptr;

  std::unique_ptr<BugType> PastEndBugType;

  void handleComparison(CheckerContext &C, const SVal &RetVal, const SVal &LVal,
                        const SVal &RVal, OverloadedOperatorKind Op) const;
  void handleAccess(CheckerContext &C, const SVal &Val) const;
  void handleDecrement(CheckerContext &C, const SVal &Val) const;
  void handleEnd(CheckerContext &C, const SVal &RetVal) const;

  bool evalFind(CheckerContext &C, const CallExpr *CE) const;
  bool evalFindEnd(CheckerContext &C, const CallExpr *CE) const;
  bool evalFindFirstOf(CheckerContext &C, const CallExpr *CE) const;
  bool evalFindIf(CheckerContext &C, const CallExpr *CE) const;
  bool evalFindIfNot(CheckerContext &C, const CallExpr *CE) const;
  bool evalLowerBound(CheckerContext &C, const CallExpr *CE) const;
  bool evalUpperBound(CheckerContext &C, const CallExpr *CE) const;
  bool evalSearch(CheckerContext &C, const CallExpr *CE) const;
  bool evalSearchN(CheckerContext &C, const CallExpr *CE) const;
  void Find(CheckerContext &C, const CallExpr *CE) const;

  void reportPastEndBug(const StringRef &Message, const SVal &Val,
                        CheckerContext &C, ExplodedNode *ErrNode) const;
  void initIdentifiers(ASTContext &Ctx) const;

public:
  IteratorPastEndChecker();

  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPreStmt(const CXXOperatorCallExpr *COCE, CheckerContext &C) const;
  void checkBeginFunction(CheckerContext &C) const;
  void checkPostStmt(const CXXConstructExpr *CCE, CheckerContext &C) const;
  void checkPostStmt(const DeclStmt *DS, CheckerContext &C) const;
  void checkPostStmt(const MaterializeTemporaryExpr *MTE,
                     CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SR, CheckerContext &C) const;
  ProgramStateRef evalAssume(ProgramStateRef State, SVal Cond,
                             bool Assumption) const;
  bool evalCall(const CallExpr *CE, CheckerContext &C) const;
};
}

REGISTER_MAP_WITH_PROGRAMSTATE(IteratorSymbolMap, SymbolRef, IteratorPosition)
REGISTER_MAP_WITH_PROGRAMSTATE(IteratorRegionMap, const MemRegion *,
                               IteratorPosition)

REGISTER_MAP_WITH_PROGRAMSTATE(IteratorComparisonMap, const SymExpr *,
                               IteratorComparison)

#define INIT_ID(Id)                                                            \
  if (!II_##Id)                                                                \
  II_##Id = &Ctx.Idents.get(#Id)

namespace {

bool isIteratorType(const QualType &Type);
bool isIterator(const CXXRecordDecl *CRD);
bool isEndCall(const FunctionDecl *Func);
bool isSimpleComparisonOperator(OverloadedOperatorKind OK);
bool isAccessOperator(OverloadedOperatorKind OK);
bool isDecrementOperator(OverloadedOperatorKind OK);
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
const IteratorPosition *getIteratorPosition(ProgramStateRef State,
                                            const SVal &Val);
const IteratorPosition *getIteratorPosition(ProgramStateRef State,
                                            RegionOrSymbol RegOrSym);
ProgramStateRef setIteratorPosition(ProgramStateRef State, const SVal &Val,
                                    IteratorPosition Pos);
ProgramStateRef setIteratorPosition(ProgramStateRef State,
                                    RegionOrSymbol RegOrSym,
                                    IteratorPosition Pos);
ProgramStateRef adjustIteratorPosition(ProgramStateRef State,
                                       RegionOrSymbol RegOrSym,
                                       IteratorPosition Pos, bool Equal);
bool contradictingIteratorPositions(IteratorPosition Pos1,
                                    IteratorPosition Pos2, bool Equal);
}

IteratorPastEndChecker::IteratorPastEndChecker() {
  PastEndBugType.reset(
      new BugType(this, "Iterator Past End", "Misuse of STL APIs"));
  PastEndBugType->setSuppressOnSink(true);
}

void IteratorPastEndChecker::checkPreCall(const CallEvent &Call,
                                          CheckerContext &C) const {
  // Check for access past end
  const auto *Func = Call.getDecl()->getAsFunction();
  if (!Func)
    return;
  if (Func->isOverloadedOperator()) {
    if (isAccessOperator(Func->getOverloadedOperator())) {
      if (const auto *InstCall = dyn_cast<CXXInstanceCall>(&Call)) {
        handleAccess(C, InstCall->getCXXThisVal());
      } else {
        handleAccess(C, Call.getArgSVal(0));
      }
    }
  }
}

void IteratorPastEndChecker::checkPostCall(const CallEvent &Call,
                                           CheckerContext &C) const {
  // Record end() iterators, iterator decrementation and comparison
  const auto *Func = Call.getDecl()->getAsFunction();
  if (!Func)
    return;
  if (Func->isOverloadedOperator()) {
    const auto Op = Func->getOverloadedOperator();
    if (isSimpleComparisonOperator(Op)) {
      if (Func->isCXXInstanceMember()) {
        const auto &InstCall = static_cast<const CXXInstanceCall &>(Call);
        handleComparison(C, InstCall.getReturnValue(), InstCall.getCXXThisVal(),
                         InstCall.getArgSVal(0), Op);
      } else {
        handleComparison(C, Call.getReturnValue(), Call.getArgSVal(0),
                         Call.getArgSVal(1), Op);
      }
    } else if (isDecrementOperator(Func->getOverloadedOperator())) {
      if (Func->isCXXInstanceMember()) {
        const auto &InstCall = static_cast<const CXXInstanceCall &>(Call);
        handleDecrement(C, InstCall.getCXXThisVal());
      } else {
        handleDecrement(C, Call.getArgSVal(0));
      }
    }
  } else if (Func->isCXXInstanceMember()) {
    if (!isEndCall(Func))
      return;
    if (!isIteratorType(Call.getResultType()))
      return;
    handleEnd(C, Call.getReturnValue());
  }
}

void IteratorPastEndChecker::checkPreStmt(const CXXOperatorCallExpr *COCE,
                                          CheckerContext &C) const {
  const auto *ThisExpr = COCE->getArg(0);

  auto State = C.getState();
  const auto *LCtx = C.getPredecessor()->getLocationContext();

  const auto CurrentThis = State->getSVal(ThisExpr, LCtx);
  if (const auto *Reg = CurrentThis.getAsRegion()) {
    if (!Reg->getAs<CXXTempObjectRegion>())
      return;
    const auto OldState = C.getPredecessor()->getFirstPred()->getState();
    const auto OldThis = OldState->getSVal(ThisExpr, LCtx);
    const auto *Pos = getIteratorPosition(OldState, OldThis);
    if (!Pos)
      return;
    State = setIteratorPosition(State, CurrentThis, *Pos);
    C.addTransition(State);
  }
}

void IteratorPastEndChecker::checkBeginFunction(CheckerContext &C) const {
  // Copy state of iterator arguments to iterator parameters
  auto State = C.getState();
  const auto *LCtx = C.getLocationContext();

  const auto *Site = cast<StackFrameContext>(LCtx)->getCallSite();
  if (!Site)
    return;

  const auto *FD = dyn_cast<FunctionDecl>(LCtx->getDecl());
  if (!FD)
    return;

  const auto *CE = dyn_cast<CallExpr>(Site);
  if (!CE)
    return;

  bool Change = false;
  int idx = 0;
  for (const auto P : FD->parameters()) {
    auto Param = State->getLValue(P, LCtx);
    auto Arg = State->getSVal(CE->getArg(idx++), LCtx->getParent());
    const auto *Pos = getIteratorPosition(State, Arg);
    if (!Pos)
      continue;
    State = setIteratorPosition(State, Param, *Pos);
    Change = true;
  }
  if (Change) {
    C.addTransition(State);
  }
}

void IteratorPastEndChecker::checkPostStmt(const CXXConstructExpr *CCE,
                                           CheckerContext &C) const {
  // Transfer iterator state in case of copy or move by constructor
  const auto *ctr = CCE->getConstructor();
  if (!ctr->isCopyOrMoveConstructor())
    return;
  const auto *RHSExpr = CCE->getArg(0);

  auto State = C.getState();
  const auto *LCtx = C.getLocationContext();

  const auto RetVal = State->getSVal(CCE, LCtx);

  const auto RHSVal = State->getSVal(RHSExpr, LCtx);
  const auto *RHSPos = getIteratorPosition(State, RHSVal);
  if (!RHSPos)
    return;
  State = setIteratorPosition(State, RetVal, *RHSPos);
  C.addTransition(State);
}

void IteratorPastEndChecker::checkPostStmt(const DeclStmt *DS,
                                           CheckerContext &C) const {
  // Transfer iterator state to new variable declaration
  for (const auto *D : DS->decls()) {
    const auto *VD = dyn_cast<VarDecl>(D);
    if (!VD || !VD->hasInit())
      continue;

    auto State = C.getState();
    const auto *LCtx = C.getPredecessor()->getLocationContext();
    const auto *Pos =
        getIteratorPosition(State, State->getSVal(VD->getInit(), LCtx));
    if (!Pos)
      continue;
    State = setIteratorPosition(State, State->getLValue(VD, LCtx), *Pos);
    C.addTransition(State);
  }
}

void IteratorPastEndChecker::checkPostStmt(const MaterializeTemporaryExpr *MTE,
                                           CheckerContext &C) const {
  /* Transfer iterator state for to temporary objects */
  auto State = C.getState();
  const auto *LCtx = C.getPredecessor()->getLocationContext();
  const auto *Pos =
      getIteratorPosition(State, State->getSVal(MTE->GetTemporaryExpr(), LCtx));
  if (!Pos)
    return;
  State = setIteratorPosition(State, State->getSVal(MTE, LCtx), *Pos);
  C.addTransition(State);
}

void IteratorPastEndChecker::checkDeadSymbols(SymbolReaper &SR,
                                              CheckerContext &C) const {
  auto State = C.getState();

  auto RegionMap = State->get<IteratorRegionMap>();
  for (const auto Reg : RegionMap) {
    if (!SR.isLiveRegion(Reg.first)) {
      State = State->remove<IteratorRegionMap>(Reg.first);
    }
  }

  auto SymbolMap = State->get<IteratorSymbolMap>();
  for (const auto Sym : SymbolMap) {
    if (SR.isDead(Sym.first)) {
      State = State->remove<IteratorSymbolMap>(Sym.first);
    }
  }

  auto ComparisonMap = State->get<IteratorComparisonMap>();
  for (const auto Comp : ComparisonMap) {
    if (SR.isDead(Comp.first)) {
      State = State->remove<IteratorComparisonMap>(Comp.first);
    }
  }
}

ProgramStateRef IteratorPastEndChecker::evalAssume(ProgramStateRef State,
                                                   SVal Cond,
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

// FIXME: Evaluation of these STL calls should be moved to StdCLibraryFunctions
//       checker (see patch r284960) or another similar checker for C++ STL
//       functions (e.g. StdCXXLibraryFunctions or StdCppLibraryFunctions).
bool IteratorPastEndChecker::evalCall(const CallExpr *CE,
                                      CheckerContext &C) const {
  const FunctionDecl *FD = C.getCalleeDecl(CE);
  if (!FD)
    return false;

  ASTContext &Ctx = C.getASTContext();
  initIdentifiers(Ctx);

  if (FD->getKind() == Decl::Function) {
    if (FD->isInStdNamespace()) {
      if (FD->getIdentifier() == II_find) {
        return evalFind(C, CE);
      } else if (FD->getIdentifier() == II_find_end) {
        return evalFindEnd(C, CE);
      } else if (FD->getIdentifier() == II_find_first_of) {
        return evalFindFirstOf(C, CE);
      } else if (FD->getIdentifier() == II_find_if) {
        return evalFindIf(C, CE);
      } else if (FD->getIdentifier() == II_find_if) {
        return evalFindIf(C, CE);
      } else if (FD->getIdentifier() == II_find_if_not) {
        return evalFindIfNot(C, CE);
      } else if (FD->getIdentifier() == II_upper_bound) {
        return evalUpperBound(C, CE);
      } else if (FD->getIdentifier() == II_lower_bound) {
        return evalLowerBound(C, CE);
      } else if (FD->getIdentifier() == II_search) {
        return evalSearch(C, CE);
      } else if (FD->getIdentifier() == II_search_n) {
        return evalSearchN(C, CE);
      }
    }
  }

  return false;
}

void IteratorPastEndChecker::handleComparison(CheckerContext &C,
                                              const SVal &RetVal,
                                              const SVal &LVal,
                                              const SVal &RVal,
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

void IteratorPastEndChecker::handleAccess(CheckerContext &C,
                                          const SVal &Val) const {
  auto State = C.getState();
  const auto *Pos = getIteratorPosition(State, Val);
  if (Pos && Pos->isOutofRange()) {
    auto *N = C.generateNonFatalErrorNode(State);
    if (!N) {
      return;
    }
    reportPastEndBug("Iterator accessed past its end.", Val, C, N);
  }
}

void IteratorPastEndChecker::handleDecrement(CheckerContext &C,
                                             const SVal &Val) const {
  auto State = C.getState();
  const auto *Pos = getIteratorPosition(State, Val);
  if (Pos && Pos->isOutofRange()) {
    State = setIteratorPosition(State, Val, IteratorPosition::getInRange());
    // FIXME: We could also check for iterators ahead of their beginnig in the
    //       future, but currently we do not care for such errors. We also
    //       assume that the iterator is not past its end by more then one
    //       position.
    C.addTransition(State);
  }
}

void IteratorPastEndChecker::handleEnd(CheckerContext &C,
                                       const SVal &RetVal) const {
  auto State = C.getState();
  State = setIteratorPosition(State, RetVal, IteratorPosition::getOutofRange());
  C.addTransition(State);
}

bool IteratorPastEndChecker::evalFind(CheckerContext &C,
                                      const CallExpr *CE) const {
  if (CE->getNumArgs() == 3 && isIteratorType(CE->getArg(0)->getType()) &&
      isIteratorType(CE->getArg(1)->getType())) {
    Find(C, CE);
    return true;
  }
  return false;
}

bool IteratorPastEndChecker::evalFindEnd(CheckerContext &C,
                                         const CallExpr *CE) const {
  if ((CE->getNumArgs() == 4 || CE->getNumArgs() == 5) &&
      isIteratorType(CE->getArg(0)->getType()) &&
      isIteratorType(CE->getArg(1)->getType()) &&
      isIteratorType(CE->getArg(2)->getType()) &&
      isIteratorType(CE->getArg(3)->getType())) {
    Find(C, CE);
    return true;
  }
  return false;
}

bool IteratorPastEndChecker::evalFindFirstOf(CheckerContext &C,
                                             const CallExpr *CE) const {
  if ((CE->getNumArgs() == 4 || CE->getNumArgs() == 5) &&
      isIteratorType(CE->getArg(0)->getType()) &&
      isIteratorType(CE->getArg(1)->getType()) &&
      isIteratorType(CE->getArg(2)->getType()) &&
      isIteratorType(CE->getArg(3)->getType())) {
    Find(C, CE);
    return true;
  }
  return false;
}

bool IteratorPastEndChecker::evalFindIf(CheckerContext &C,
                                        const CallExpr *CE) const {
  if (CE->getNumArgs() == 3 && isIteratorType(CE->getArg(0)->getType()) &&
      isIteratorType(CE->getArg(1)->getType())) {
    Find(C, CE);
    return true;
  }
  return false;
}

bool IteratorPastEndChecker::evalFindIfNot(CheckerContext &C,
                                           const CallExpr *CE) const {
  if (CE->getNumArgs() == 3 && isIteratorType(CE->getArg(0)->getType()) &&
      isIteratorType(CE->getArg(1)->getType())) {
    Find(C, CE);
    return true;
  }
  return false;
}

bool IteratorPastEndChecker::evalLowerBound(CheckerContext &C,
                                            const CallExpr *CE) const {
  if ((CE->getNumArgs() == 3 || CE->getNumArgs() == 4) &&
      isIteratorType(CE->getArg(0)->getType()) &&
      isIteratorType(CE->getArg(1)->getType())) {
    Find(C, CE);
    return true;
  }
  return false;
}

bool IteratorPastEndChecker::evalUpperBound(CheckerContext &C,
                                            const CallExpr *CE) const {
  if ((CE->getNumArgs() == 3 || CE->getNumArgs() == 4) &&
      isIteratorType(CE->getArg(0)->getType()) &&
      isIteratorType(CE->getArg(1)->getType())) {
    Find(C, CE);
    return true;
  }
  return false;
}

bool IteratorPastEndChecker::evalSearch(CheckerContext &C,
                                        const CallExpr *CE) const {
  if ((CE->getNumArgs() == 4 || CE->getNumArgs() == 5) &&
      isIteratorType(CE->getArg(0)->getType()) &&
      isIteratorType(CE->getArg(1)->getType()) &&
      isIteratorType(CE->getArg(2)->getType()) &&
      isIteratorType(CE->getArg(3)->getType())) {
    Find(C, CE);
    return true;
  }
  return false;
}

bool IteratorPastEndChecker::evalSearchN(CheckerContext &C,
                                         const CallExpr *CE) const {
  if ((CE->getNumArgs() == 4 || CE->getNumArgs() == 5) &&
      isIteratorType(CE->getArg(0)->getType()) &&
      isIteratorType(CE->getArg(1)->getType())) {
    Find(C, CE);
    return true;
  }
  return false;
}

void IteratorPastEndChecker::Find(CheckerContext &C, const CallExpr *CE) const {
  auto state = C.getState();
  auto &svalBuilder = C.getSValBuilder();
  const auto *LCtx = C.getLocationContext();

  auto RetVal = svalBuilder.conjureSymbolVal(nullptr, CE, LCtx, C.blockCount());
  auto SecondParam = state->getSVal(CE->getArg(1), LCtx);

  auto stateFound = state->BindExpr(CE, LCtx, RetVal);
  auto stateNotFound = state->BindExpr(CE, LCtx, SecondParam);

  C.addTransition(stateFound);
  C.addTransition(stateNotFound);
}

void IteratorPastEndChecker::reportPastEndBug(const StringRef &Message,
                                              const SVal &Val,
                                              CheckerContext &C,
                                              ExplodedNode *ErrNode) const {
  auto R = llvm::make_unique<BugReport>(*PastEndBugType, Message, ErrNode);
  R->markInteresting(Val);
  C.emitReport(std::move(R));
}

void IteratorPastEndChecker::initIdentifiers(ASTContext &Ctx) const {
  INIT_ID(find);
  INIT_ID(find_end);
  INIT_ID(find_first_of);
  INIT_ID(find_if);
  INIT_ID(find_if_not);
  INIT_ID(lower_bound);
  INIT_ID(upper_bound);
  INIT_ID(search);
  INIT_ID(search_n);
}

namespace {

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

bool isEndCall(const FunctionDecl *Func) {
  const auto *IdInfo = Func->getIdentifier();
  if (!IdInfo)
    return false;
  return IdInfo->getName().endswith_lower("end");
}

bool isSimpleComparisonOperator(OverloadedOperatorKind OK) {
  return OK == OO_EqualEqual || OK == OO_ExclaimEqual;
}

bool isAccessOperator(OverloadedOperatorKind OK) {
  return OK == OO_Star || OK == OO_Arrow || OK == OO_ArrowStar ||
         OK == OO_Plus || OK == OO_PlusEqual || OK == OO_PlusPlus ||
         OK == OO_Subscript;
}

bool isDecrementOperator(OverloadedOperatorKind OK) {
  return OK == OO_MinusEqual || OK == OO_MinusMinus;
}

BinaryOperator::Opcode getOpcode(const SymExpr *SE) {
  if (const auto *BSE = dyn_cast<BinarySymExpr>(SE)) {
    return BSE->getOpcode();
  } else if (const auto *SC = dyn_cast<SymbolConjured>(SE)) {
    const auto *COE = dyn_cast<CXXOperatorCallExpr>(SC->getStmt());
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
    if (contradictingIteratorPositions(*LPos, *RPos, Equal)) {
      return nullptr;
    }
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
                                    IteratorPosition Pos) {
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
                                    IteratorPosition Pos) {
  if (RegOrSym.is<const MemRegion *>()) {
    return State->set<IteratorRegionMap>(RegOrSym.get<const MemRegion *>(),
                                         Pos);
  } else if (RegOrSym.is<SymbolRef>()) {
    return State->set<IteratorSymbolMap>(RegOrSym.get<SymbolRef>(), Pos);
  }
  return nullptr;
}

ProgramStateRef adjustIteratorPosition(ProgramStateRef State,
                                       RegionOrSymbol RegOrSym,
                                       IteratorPosition Pos, bool Equal) {

  if ((Pos.isInRange() && Equal) || (Pos.isOutofRange() && !Equal)) {
    return setIteratorPosition(State, RegOrSym, IteratorPosition::getInRange());
  } else if (Pos.isOutofRange() && Equal) {
    return setIteratorPosition(State, RegOrSym,
                               IteratorPosition::getOutofRange());
  } else {
    return State;
  }
}

bool contradictingIteratorPositions(IteratorPosition Pos1,
                                    IteratorPosition Pos2, bool Equal) {
  return ((Pos1 != Pos2) && Equal) ||
         ((Pos1.isOutofRange() && Pos2.isOutofRange()) && !Equal);
}
}

void ento::registerIteratorPastEndChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<IteratorPastEndChecker>();
}
