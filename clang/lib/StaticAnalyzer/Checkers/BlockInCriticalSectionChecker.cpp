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
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

#include <iterator>
#include <utility>
#include <variant>

using namespace clang;
using namespace ento;

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

class FirstArgMutexDescriptor {
  CallDescription LockFn;
  CallDescription UnlockFn;

public:
  FirstArgMutexDescriptor(CallDescription &&LockFn, CallDescription &&UnlockFn)
      : LockFn(std::move(LockFn)), UnlockFn(std::move(UnlockFn)) {}
  [[nodiscard]] bool matchesLock(const CallEvent &Call) const {
    return LockFn.matches(Call) && Call.getNumArgs() > 0;
  }
  [[nodiscard]] bool matchesUnlock(const CallEvent &Call) const {
    return UnlockFn.matches(Call) && Call.getNumArgs() > 0;
  }
  [[nodiscard]] const MemRegion *getLockRegion(const CallEvent &Call) const {
    return Call.getArgSVal(0).getAsRegion();
  }
  [[nodiscard]] const MemRegion *getUnlockRegion(const CallEvent &Call) const {
    return Call.getArgSVal(0).getAsRegion();
  }
};

class MemberMutexDescriptor {
  CallDescription LockFn;
  CallDescription UnlockFn;

public:
  MemberMutexDescriptor(CallDescription &&LockFn, CallDescription &&UnlockFn)
      : LockFn(std::move(LockFn)), UnlockFn(std::move(UnlockFn)) {}
  [[nodiscard]] bool matchesLock(const CallEvent &Call) const {
    return LockFn.matches(Call);
  }
  bool matchesUnlock(const CallEvent &Call) const {
    return UnlockFn.matches(Call);
  }
  [[nodiscard]] const MemRegion *getLockRegion(const CallEvent &Call) const {
    return cast<CXXMemberCall>(Call).getCXXThisVal().getAsRegion();
  }
  [[nodiscard]] const MemRegion *getUnlockRegion(const CallEvent &Call) const {
    return cast<CXXMemberCall>(Call).getCXXThisVal().getAsRegion();
  }
};

class RAIIMutexDescriptor {
  mutable const IdentifierInfo *Guard{};
  mutable bool IdentifierInfoInitialized{};
  mutable llvm::SmallString<32> GuardName{};

  void initIdentifierInfo(const CallEvent &Call) const {
    if (!IdentifierInfoInitialized) {
      // In case of checking C code, or when the corresponding headers are not
      // included, we might end up query the identifier table every time when
      // this function is called instead of early returning it. To avoid this, a
      // bool variable (IdentifierInfoInitialized) is used and the function will
      // be run only once.
      Guard = &Call.getCalleeAnalysisDeclContext()->getASTContext().Idents.get(
          GuardName);
      IdentifierInfoInitialized = true;
    }
  }

public:
  RAIIMutexDescriptor(StringRef GuardName) : GuardName(GuardName) {}
  [[nodiscard]] bool matchesLock(const CallEvent &Call) const {
    initIdentifierInfo(Call);
    const auto *Ctor = dyn_cast<CXXConstructorCall>(&Call);
    if (!Ctor)
      return false;
    auto *IdentifierInfo = Ctor->getDecl()->getParent()->getIdentifier();
    return IdentifierInfo == Guard;
  }
  [[nodiscard]] bool matchesUnlock(const CallEvent &Call) const {
    initIdentifierInfo(Call);
    const auto *Dtor = dyn_cast<CXXDestructorCall>(&Call);
    if (!Dtor)
      return false;
    auto *IdentifierInfo =
        cast<CXXRecordDecl>(Dtor->getDecl()->getParent())->getIdentifier();
    return IdentifierInfo == Guard;
  }
  [[nodiscard]] const MemRegion *getLockRegion(const CallEvent &Call) const {
    const MemRegion *LockRegion = nullptr;
    if (std::optional<SVal> Object = Call.getReturnValueUnderConstruction()) {
      LockRegion = Object->getAsRegion();
    }
    return LockRegion;
  }
  [[nodiscard]] const MemRegion *getUnlockRegion(const CallEvent &Call) const {
    return cast<CXXDestructorCall>(Call).getCXXThisVal().getAsRegion();
  }
};

using MutexDescriptor =
    std::variant<FirstArgMutexDescriptor, MemberMutexDescriptor,
                 RAIIMutexDescriptor>;

class BlockInCriticalSectionChecker : public Checker<check::PostCall> {
private:
  const std::array<MutexDescriptor, 8> MutexDescriptors{
      MemberMutexDescriptor(
          CallDescription(/*QualifiedName=*/{"std", "mutex", "lock"},
                          /*RequiredArgs=*/0),
          CallDescription({"std", "mutex", "unlock"}, 0)),
      FirstArgMutexDescriptor(CallDescription({"pthread_mutex_lock"}, 1),
                              CallDescription({"pthread_mutex_unlock"}, 1)),
      FirstArgMutexDescriptor(CallDescription({"mtx_lock"}, 1),
                              CallDescription({"mtx_unlock"}, 1)),
      FirstArgMutexDescriptor(CallDescription({"pthread_mutex_trylock"}, 1),
                              CallDescription({"pthread_mutex_unlock"}, 1)),
      FirstArgMutexDescriptor(CallDescription({"mtx_trylock"}, 1),
                              CallDescription({"mtx_unlock"}, 1)),
      FirstArgMutexDescriptor(CallDescription({"mtx_timedlock"}, 1),
                              CallDescription({"mtx_unlock"}, 1)),
      RAIIMutexDescriptor("lock_guard"),
      RAIIMutexDescriptor("unique_lock")};

  const std::array<CallDescription, 5> BlockingFunctions{
      ArrayRef{StringRef{"sleep"}}, ArrayRef{StringRef{"getc"}},
      ArrayRef{StringRef{"fgets"}}, ArrayRef{StringRef{"read"}},
      ArrayRef{StringRef{"recv"}}};

  const BugType BlockInCritSectionBugType{
      this, "Call to blocking function in critical section", "Blocking Error"};

  void reportBlockInCritSection(const CallEvent &call, CheckerContext &C) const;

  [[nodiscard]] const NoteTag *createCritSectionNote(CritSectionMarker M,
                                                     CheckerContext &C) const;
  [[nodiscard]] std::optional<MutexDescriptor>
  checkLock(const CallEvent &Call, CheckerContext &C) const;
  void handleLock(const MutexDescriptor &Mutex, const CallEvent &Call,
                  CheckerContext &C) const;
  [[nodiscard]] std::optional<MutexDescriptor>
  checkUnlock(const CallEvent &Call, CheckerContext &C) const;
  void handleUnlock(const MutexDescriptor &Mutex, const CallEvent &Call,
                    CheckerContext &C) const;
  [[nodiscard]] bool isBlockingInCritSection(const CallEvent &Call,
                                             CheckerContext &C) const;

public:
  /// Process unlock.
  /// Process lock.
  /// Process blocking functions (sleep, getc, fgets, read, recv)
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
};

} // end anonymous namespace

REGISTER_LIST_WITH_PROGRAMSTATE(ActiveCritSections, CritSectionMarker)

namespace std {
// Iterator traits for ImmutableList data structure
// that enable the use of STL algorithms.
// TODO: Move these to llvm::ImmutableList when overhauling immutable data
// structures for proper iterator concept support.
template <>
struct iterator_traits<
    typename llvm::ImmutableList<CritSectionMarker>::iterator> {
  using iterator_category = std::forward_iterator_tag;
  using value_type = CritSectionMarker;
  using difference_type = std::ptrdiff_t;
  using reference = CritSectionMarker &;
  using pointer = CritSectionMarker *;
};
} // namespace std

std::optional<MutexDescriptor>
BlockInCriticalSectionChecker::checkLock(const CallEvent &Call,
                                         CheckerContext &C) const {
  const auto *LockDescriptor =
      llvm::find_if(MutexDescriptors, [&Call](auto &&LockFn) {
        return std::visit(
            [&Call](auto &&Descriptor) { return Descriptor.matchesLock(Call); },
            LockFn);
      });
  if (LockDescriptor != MutexDescriptors.end())
    return *LockDescriptor;
  return std::nullopt;
}

void BlockInCriticalSectionChecker::handleLock(
    const MutexDescriptor &LockDescriptor, const CallEvent &Call,
    CheckerContext &C) const {
  const auto *MutexRegion = std::visit(
      [&Call](auto &&Descriptor) { return Descriptor.getLockRegion(Call); },
      LockDescriptor);
  if (!MutexRegion)
    return;

  const auto MarkToAdd = CritSectionMarker{Call.getOriginExpr(), MutexRegion};
  ProgramStateRef StateWithLockEvent =
      C.getState()->add<ActiveCritSections>(MarkToAdd);
  C.addTransition(StateWithLockEvent, createCritSectionNote(MarkToAdd, C));
}

std::optional<MutexDescriptor>
BlockInCriticalSectionChecker::checkUnlock(const CallEvent &Call,
                                           CheckerContext &C) const {
  const auto *UnlockDescriptor =
      llvm::find_if(MutexDescriptors, [&Call](auto &&UnlockFn) {
        return std::visit(
            [&Call](auto &&Descriptor) {
              return Descriptor.matchesUnlock(Call);
            },
            UnlockFn);
      });
  if (UnlockDescriptor != MutexDescriptors.end())
    return *UnlockDescriptor;
  return std::nullopt;
}

void BlockInCriticalSectionChecker::handleUnlock(
    const MutexDescriptor &UnlockDescriptor, const CallEvent &Call,
    CheckerContext &C) const {
  const auto *MutexRegion = std::visit(
      [&Call](auto &&Descriptor) { return Descriptor.getUnlockRegion(Call); },
      UnlockDescriptor);
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
  return llvm::any_of(BlockingFunctions,
                      [&Call](auto &&Fn) { return Fn.matches(Call); }) &&
         !C.getState()->get<ActiveCritSections>().isEmpty();
}

void BlockInCriticalSectionChecker::checkPostCall(const CallEvent &Call,
                                                  CheckerContext &C) const {
  if (isBlockingInCritSection(Call, C)) {
    reportBlockInCritSection(Call, C);
  } else if (auto Lock = checkLock(Call, C)) {
    handleLock(*Lock, Call, C);
  } else if (auto Unlock = checkUnlock(Call, C)) {
    handleUnlock(*Unlock, Call, C);
  }
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
    const auto *Position =
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
