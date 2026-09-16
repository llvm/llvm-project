//===-- BlockInCriticalSectionChecker.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a checker for blocks in critical sections. This checker should find
// the calls to blocking functions (for example: sleep, getc, fgets, read,
// recv etc.) inside a critical section. When sleep(x) is called while a mutex
// is held, other threades cannot lock the same mutex. This might take some
// time, leading to bad performance or even deadlock.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

#include <iterator>
#include <utility>

using namespace clang;
using namespace ento;

static const MemRegion *getFirstArgRegion(const CallEvent &Call) {
  return Call.getArgSVal(0).getAsRegion();
}

static const MemRegion *getCXXThisRegion(const CallEvent &Call) {
  return cast<CXXMemberCall>(Call).getCXXThisVal().getAsRegion();
}

static const MemRegion *getObjectUnderConstruction(const CallEvent &Call) {
  if (std::optional<SVal> Object = Call.getReturnValueUnderConstruction())
    return Object->getAsRegion();
  return nullptr;
}

static const MemRegion *getCXXDestructorThisRegion(const CallEvent &Call) {
  return cast<CXXDestructorCall>(Call).getCXXThisVal().getAsRegion();
}

static bool isNotDeferLockUniqueLock(const CallEvent &Call) {
  if (Call.getNumArgs() < 2)
    return true;
  const Expr *SecondArg = Call.getArgExpr(1);
  QualType ArgType = SecondArg->getType().getNonReferenceType();
  if (const auto *RD = ArgType->getAsRecordDecl();
      RD && RD->getName() == "defer_lock_t" && RD->isInStdNamespace())
    return false;
  return true;
}

namespace {

struct CritSectionMarker {
  const Expr *LockExpr{};
  const MemRegion *LockReg{};

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.Add(LockExpr);
    ID.Add(LockReg);
  }

  [[nodiscard]] constexpr bool
  operator==(const CritSectionMarker &Other) const noexcept {
    return LockExpr == Other.LockExpr && LockReg == Other.LockReg;
  }
  [[nodiscard]] constexpr bool
  operator!=(const CritSectionMarker &Other) const noexcept {
    return !(*this == Other);
  }
};

enum class Role {
  Lock,
  Unlock,
};

using GetRegionFn = const MemRegion *(*)(const CallEvent &);
using FilterFn = bool (*)(const CallEvent &);

struct ThreadingCallDescription {
  Role Role;
  GetRegionFn GetRegion = getFirstArgRegion;
  FilterFn Filter = nullptr;
};

class SuppressNonBlockingStreams : public BugReporterVisitor {
private:
  const CallDescription OpenFunction{CDM::CLibrary, {"open"}, 2};
  SymbolRef StreamSym;
  const int NonBlockMacroVal;
  bool Satisfied = false;

public:
  SuppressNonBlockingStreams(SymbolRef StreamSym, int NonBlockMacroVal)
      : StreamSym(StreamSym), NonBlockMacroVal(NonBlockMacroVal) {}

  static void *getTag() {
    static bool Tag;
    return &Tag;
  }

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    ID.AddPointer(getTag());
  }

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override {
    if (Satisfied)
      return nullptr;

    std::optional<StmtPoint> Point = N->getLocationAs<StmtPoint>();
    if (!Point)
      return nullptr;

    const auto *CE = Point->getStmtAs<CallExpr>();
    if (!CE || !OpenFunction.matchesAsWritten(*CE))
      return nullptr;

    if (N->getSVal(CE).getAsSymbol() != StreamSym)
      return nullptr;

    Satisfied = true;

    // Check if open's second argument contains O_NONBLOCK
    const llvm::APSInt *FlagVal = N->getSVal(CE->getArg(1)).getAsInteger();
    if (!FlagVal)
      return nullptr;

    if ((*FlagVal & NonBlockMacroVal) != 0)
      BR.markInvalid(getTag(), nullptr);

    return nullptr;
  }
};

class BlockInCriticalSectionChecker
    : public Checker<check::PostCall, eval::Call> {
private:
  const CallDescriptionMap<ThreadingCallDescription> ThreadingCalls{
      // NOTE: There are standard library implementations where some methods
      // of `std::mutex` are inherited from an implementation detail base
      // class, and those aren't matched by the name specification {"std",
      // "mutex", "lock"}.
      // As a workaround here we omit the class name and only require the
      // presence of the name parts "std" and "lock"/"unlock".
      // TODO: Ensure that CallDescription understands inherited methods.
      {{CDM::CXXMethod, {"std", /*"mutex",*/ "lock"}, 0},
       {Role::Lock, getCXXThisRegion}},
      {{CDM::CXXMethod, {"std", /*"mutex",*/ "unlock"}, 0},
       {Role::Unlock, getCXXThisRegion}},
      {{CDM::CLibrary, {"pthread_mutex_lock"}, 1}, {Role::Lock}},
      {{CDM::CLibrary, {"pthread_mutex_unlock"}, 1}, {Role::Unlock}},
      {{CDM::CLibrary, {"mtx_lock"}, 1}, {Role::Lock}},
      {{CDM::CLibrary, {"mtx_unlock"}, 1}, {Role::Unlock}},
      {{CDM::CLibrary, {"pthread_mutex_trylock"}, 1}, {Role::Lock}},
      {{CDM::CLibrary, {"mtx_trylock"}, 1}, {Role::Lock}},
      {{CDM::CLibrary, {"mtx_timedlock"}, 1}, {Role::Lock}},
      {{CDM::CXXMethod, {"lock_guard", "lock_guard"}},
       {Role::Lock, getObjectUnderConstruction}},
      {{CDM::CXXMethod, {"lock_guard", "~lock_guard"}},
       {Role::Unlock, getCXXDestructorThisRegion}},
      {{CDM::CXXMethod, {"unique_lock", "unique_lock"}},
       {Role::Lock, getObjectUnderConstruction, isNotDeferLockUniqueLock}},
      {{CDM::CXXMethod, {"unique_lock", "~unique_lock"}},
       {Role::Unlock, getCXXDestructorThisRegion}},
      {{CDM::CXXMethod, {"scoped_lock", "scoped_lock"}},
       {Role::Lock, getObjectUnderConstruction}},
      {{CDM::CXXMethod, {"scoped_lock", "~scoped_lock"}},
       {Role::Unlock, getCXXDestructorThisRegion}},
  };

  const CallDescriptionSet BlockingFunctions{{CDM::CLibrary, {"sleep"}},
                                             {CDM::CLibrary, {"getc"}},
                                             {CDM::CLibrary, {"fgets"}},
                                             {CDM::CLibrary, {"read"}},
                                             {CDM::CLibrary, {"recv"}}};

  const BugType BlockInCritSectionBugType{
      this, "Call to blocking function in critical section", "Blocking Error"};

  using O_NONBLOCKValueTy = std::optional<int>;
  mutable std::optional<O_NONBLOCKValueTy> O_NONBLOCKValue;

  void reportBlockInCritSection(const CallEvent &call, CheckerContext &C) const;

  [[nodiscard]] const NoteTag *createCritSectionNote(CritSectionMarker M,
                                                     CheckerContext &C) const;

  [[nodiscard]] const ThreadingCallDescription *
  lookupThreadingCall(const CallEvent &Call) const;

  void handleLock(const ThreadingCallDescription &Desc, const CallEvent &Call,
                  CheckerContext &C, ProgramStateRef State) const;

  void handleUnlock(const ThreadingCallDescription &Desc, const CallEvent &Call,
                    CheckerContext &C) const;

  [[nodiscard]] bool isBlockingInCritSection(const CallEvent &Call,
                                             CheckerContext &C) const;

public:
  /// Process unlock.
  /// Process lock.
  /// Process blocking functions (sleep, getc, fgets, read, recv)
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;

  // Process RAII lock guard constructors (to avoid double-counting by
  // inlining).
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
};

} // end anonymous namespace

REGISTER_LIST_WITH_PROGRAMSTATE(ActiveCritSections, CritSectionMarker)

const ThreadingCallDescription *
BlockInCriticalSectionChecker::lookupThreadingCall(
    const CallEvent &Call) const {
  const ThreadingCallDescription *Desc = ThreadingCalls.lookup(Call);
  if (!Desc)
    return nullptr;
  if (Desc->Filter && !Desc->Filter(Call))
    return nullptr;
  return Desc;
}

static const MemRegion *skipStdBaseClassRegion(const MemRegion *Reg) {
  while (Reg) {
    const auto *BaseClassRegion = dyn_cast<CXXBaseObjectRegion>(Reg);
    if (!BaseClassRegion || !isWithinStdNamespace(BaseClassRegion->getDecl()))
      break;
    Reg = BaseClassRegion->getSuperRegion();
  }
  return Reg;
}

static const MemRegion *getMutexRegion(const CallEvent &Call,
                                       const ThreadingCallDescription &Desc) {
  return skipStdBaseClassRegion(Desc.GetRegion(Call));
}

void BlockInCriticalSectionChecker::handleLock(
    const ThreadingCallDescription &Desc, const CallEvent &Call,
    CheckerContext &C, ProgramStateRef State) const {
  const MemRegion *MutexRegion = getMutexRegion(Call, Desc);
  if (!MutexRegion)
    return;

  const CritSectionMarker MarkToAdd{Call.getOriginExpr(), MutexRegion};
  ProgramStateRef StateWithLockEvent =
      State->add<ActiveCritSections>(MarkToAdd);
  C.addTransition(StateWithLockEvent, createCritSectionNote(MarkToAdd, C));
}

void BlockInCriticalSectionChecker::handleUnlock(
    const ThreadingCallDescription &Desc, const CallEvent &Call,
    CheckerContext &C) const {
  const MemRegion *MutexRegion = getMutexRegion(Call, Desc);
  if (!MutexRegion)
    return;

  ProgramStateRef State = C.getState();
  const auto ActiveSections = State->get<ActiveCritSections>();
  const auto MostRecentLock =
      llvm::find_if(ActiveSections, [MutexRegion](auto &&Marker) {
        return Marker.LockReg == MutexRegion;
      });
  if (MostRecentLock == ActiveSections.end())
    return;

  // Build a new ImmutableList without this element.
  auto &Factory = State->get_context<ActiveCritSections>();
  llvm::ImmutableList<CritSectionMarker> NewList = Factory.getEmptyList();
  for (auto It = ActiveSections.begin(), End = ActiveSections.end(); It != End;
       ++It) {
    if (It != MostRecentLock)
      NewList = Factory.add(*It, NewList);
  }

  State = State->set<ActiveCritSections>(NewList);
  C.addTransition(State);
}

bool BlockInCriticalSectionChecker::isBlockingInCritSection(
    const CallEvent &Call, CheckerContext &C) const {
  return BlockingFunctions.contains(Call) &&
         !C.getState()->get<ActiveCritSections>().isEmpty();
}

void BlockInCriticalSectionChecker::checkPostCall(const CallEvent &Call,
                                                  CheckerContext &C) const {
  if (isBlockingInCritSection(Call, C)) {
    reportBlockInCritSection(Call, C);
    return;
  }

  const ThreadingCallDescription *Desc = lookupThreadingCall(Call);
  if (!Desc)
    return;

  // RAII constructors are modeled in evalCall so they are not inlined.
  if (isa<CXXConstructorCall>(Call))
    return;

  switch (Desc->Role) {
  case Role::Lock:
    handleLock(*Desc, Call, C, C.getState());
    break;
  case Role::Unlock:
    handleUnlock(*Desc, Call, C);
    break;
  }
}

bool BlockInCriticalSectionChecker::evalCall(const CallEvent &Call,
                                             CheckerContext &C) const {
  const ThreadingCallDescription *Desc = lookupThreadingCall(Call);
  if (!Desc || !isa<CXXConstructorCall>(Call))
    return false;

  ProgramStateRef State = C.getState();
  // Escape the object under construction to model the side-effects of the
  // constructor.
  if (const auto *Ctor = dyn_cast<AnyCXXConstructorCall>(&Call)) {
    const MemRegion *ObjRegion = Ctor->getCXXThisVal().getAsRegion();
    State = State->invalidateRegions(ObjRegion, C.getCFGElementRef(),
                                     C.blockCount(), C.getStackFrame(),
                                     /*CausesPointerEscape=*/false);
  }
  handleLock(*Desc, Call, C, State);
  return true;
}

void BlockInCriticalSectionChecker::reportBlockInCritSection(
    const CallEvent &Call, CheckerContext &C) const {
  ExplodedNode *ErrNode = C.generateNonFatalErrorNode(C.getState());
  if (!ErrNode)
    return;

  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "Call to blocking function '" << Call.getCalleeIdentifier()->getName()
     << "' inside of critical section";
  auto R = std::make_unique<PathSensitiveBugReport>(BlockInCritSectionBugType,
                                                    os.str(), ErrNode);
  // for 'read' and 'recv' call, check whether it's file descriptor(first
  // argument) is
  // created by 'open' API with O_NONBLOCK flag or is equal to -1, they will
  // not cause block in these situations, don't report
  StringRef FuncName = Call.getCalleeIdentifier()->getName();
  if (FuncName == "read" || FuncName == "recv") {
    SVal SV = Call.getArgSVal(0);
    SValBuilder &SVB = C.getSValBuilder();
    ProgramStateRef state = C.getState();
    ConditionTruthVal CTV =
        state->areEqual(SV, SVB.makeIntVal(-1, C.getASTContext().IntTy));
    if (CTV.isConstrainedTrue())
      return;

    if (SymbolRef SR = SV.getAsSymbol()) {
      if (!O_NONBLOCKValue)
        O_NONBLOCKValue = tryExpandAsInteger(
            "O_NONBLOCK", C.getBugReporter().getPreprocessor());
      if (*O_NONBLOCKValue)
        R->addVisitor<SuppressNonBlockingStreams>(SR, **O_NONBLOCKValue);
    }
  }
  R->addRange(Call.getSourceRange());
  R->markInteresting(Call.getReturnValue());
  C.emitReport(std::move(R));
}

const NoteTag *
BlockInCriticalSectionChecker::createCritSectionNote(CritSectionMarker M,
                                                     CheckerContext &C) const {
  const BugType *BT = &this->BlockInCritSectionBugType;
  return C.getNoteTag([M, BT](PathSensitiveBugReport &BR,
                              llvm::raw_ostream &OS) {
    if (&BR.getBugType() != BT)
      return;

    // Get the lock events for the mutex of the current line's lock event.
    const auto CritSectionBegins =
        BR.getErrorNode()->getState()->get<ActiveCritSections>();
    llvm::SmallVector<CritSectionMarker, 4> LocksForMutex;
    llvm::copy_if(
        CritSectionBegins, std::back_inserter(LocksForMutex),
        [M](const auto &Marker) { return Marker.LockReg == M.LockReg; });
    if (LocksForMutex.empty())
      return;

    // As the ImmutableList builds the locks by prepending them, we
    // reverse the list to get the correct order.
    std::reverse(LocksForMutex.begin(), LocksForMutex.end());

    // Find the index of the lock expression in the list of all locks for a
    // given mutex (in acquisition order).
    const auto Position =
        llvm::find_if(std::as_const(LocksForMutex), [M](const auto &Marker) {
          return Marker.LockExpr == M.LockExpr;
        });
    if (Position == LocksForMutex.end())
      return;

    // If there is only one lock event, we don't need to specify how many times
    // the critical section was entered.
    if (LocksForMutex.size() == 1) {
      OS << "Entering critical section here";
      return;
    }

    const auto IndexOfLock =
        std::distance(std::as_const(LocksForMutex).begin(), Position);

    const auto OrdinalOfLock = IndexOfLock + 1;
    OS << "Entering critical section for the " << OrdinalOfLock
       << llvm::getOrdinalSuffix(OrdinalOfLock) << " time here";
  });
}

void ento::registerBlockInCriticalSectionChecker(CheckerManager &mgr) {
  mgr.registerChecker<BlockInCriticalSectionChecker>();
}

bool ento::shouldRegisterBlockInCriticalSectionChecker(
    const CheckerManager &mgr) {
  return true;
}
