//===-- SimpleStreamChecker.cpp -----------------------------------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines a checker for proper use of fopen/fclose APIs.
//   - If a file has been closed with fclose, it should not be accessed again.
//   Accessing a closed file results in undefined behavior.
//   - If a file was opened with fopen, it must be closed with fclose before
//   the execution ends. Failing to do so results in a resource leak.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
typedef llvm::SmallVector<SymbolRef, 2> SymbolVector;

struct StreamState {
private:
  enum Kind { Opened, Closed } K;
  StreamState(Kind InK) : K(InK) { }

public:
  bool isOpened() const { return K == Opened; }
  bool isClosed() const { return K == Closed; }

  static StreamState getOpened() { return StreamState(Opened); }
  static StreamState getClosed() { return StreamState(Closed); }

  bool operator==(const StreamState &X) const {
    return K == X.K;
  }
  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(K);
  }
};

class SimpleStreamChecker : public Checker<check::PostCall,
                                           check::PreCall,
                                           check::DeadSymbols,
                                           check::Bind,
                                           check::RegionChanges> {

  mutable IdentifierInfo *IIfopen, *IIfclose;

  OwningPtr<BugType> DoubleCloseBugType;
  OwningPtr<BugType> LeakBugType;

  void initIdentifierInfo(ASTContext &Ctx) const;

  void reportDoubleClose(SymbolRef FileDescSym,
                         const CallEvent &Call,
                         CheckerContext &C) const;

  void reportLeaks(SymbolVector LeakedStreams,
                   CheckerContext &C,
                   ExplodedNode *ErrNode) const;

  bool guaranteedNotToCloseFile(const CallEvent &Call) const;

public:
  SimpleStreamChecker();

  /// Process fopen.
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  /// Process fclose.
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;

  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;

  /// Deal with symbol escape as a byproduct of a bind.
  void checkBind(SVal location, SVal val, const Stmt*S,
                 CheckerContext &C) const;

  /// Deal with symbol escape as a byproduct of a region change.
  ProgramStateRef
  checkRegionChanges(ProgramStateRef state,
                     const StoreManager::InvalidatedSymbols *invalidated,
                     ArrayRef<const MemRegion *> ExplicitRegions,
                     ArrayRef<const MemRegion *> Regions,
                     const CallEvent *Call) const;
  bool wantsRegionChangeUpdate(ProgramStateRef state) const {
    return true;
  }
};

} // end anonymous namespace

/// The state of the checker is a map from tracked stream symbols to their
/// state. Let's store it in the ProgramState.
REGISTER_MAP_WITH_PROGRAMSTATE(StreamMap, SymbolRef, StreamState)

namespace {
class StopTrackingCallback : public SymbolVisitor {
  ProgramStateRef state;
public:
  StopTrackingCallback(ProgramStateRef st) : state(st) {}
  ProgramStateRef getState() const { return state; }

  bool VisitSymbol(SymbolRef sym) {
    state = state->remove<StreamMap>(sym);
    return true;
  }
};
} // end anonymous namespace


SimpleStreamChecker::SimpleStreamChecker() : IIfopen(0), IIfclose(0) {
  // Initialize the bug types.
  DoubleCloseBugType.reset(new BugType("Double fclose",
                                       "Unix Stream API Error"));

  LeakBugType.reset(new BugType("Resource Leak",
                                "Unix Stream API Error"));
  // Sinks are higher importance bugs as well as calls to assert() or exit(0).
  LeakBugType->setSuppressOnSink(true);
}

void SimpleStreamChecker::checkPostCall(const CallEvent &Call,
                                        CheckerContext &C) const {
  initIdentifierInfo(C.getASTContext());

  if (!Call.isGlobalCFunction())
    return;

  if (Call.getCalleeIdentifier() != IIfopen)
    return;

  // Get the symbolic value corresponding to the file handle.
  SymbolRef FileDesc = Call.getReturnValue().getAsSymbol();
  if (!FileDesc)
    return;

  // Generate the next transition (an edge in the exploded graph).
  ProgramStateRef State = C.getState();
  State = State->set<StreamMap>(FileDesc, StreamState::getOpened());
  C.addTransition(State);
}

void SimpleStreamChecker::checkPreCall(const CallEvent &Call,
                                       CheckerContext &C) const {
  initIdentifierInfo(C.getASTContext());

  if (!Call.isGlobalCFunction())
    return;

  if (Call.getCalleeIdentifier() != IIfclose)
    return;

  if (Call.getNumArgs() != 1)
    return;

  // Get the symbolic value corresponding to the file handle.
  SymbolRef FileDesc = Call.getArgSVal(0).getAsSymbol();
  if (!FileDesc)
    return;

  // Check if the stream has already been closed.
  ProgramStateRef State = C.getState();
  const StreamState *SS = State->get<StreamMap>(FileDesc);
  if (SS && SS->isClosed()) {
    reportDoubleClose(FileDesc, Call, C);
    return;
  }

  // Generate the next transition, in which the stream is closed.
  State = State->set<StreamMap>(FileDesc, StreamState::getClosed());
  C.addTransition(State);
}

static bool isLeaked(SymbolRef Sym, const StreamState &SS,
                     bool IsSymDead, ProgramStateRef State) {
  if (IsSymDead && SS.isOpened()) {
    // If a symbol is NULL, assume that fopen failed on this path.
    // A symbol should only be considered leaked if it is non-null.
    ConstraintManager &CMgr = State->getConstraintManager();
    ConditionTruthVal OpenFailed = CMgr.isNull(State, Sym);
    return !OpenFailed.isConstrainedTrue();
  }
  return false;
}

void SimpleStreamChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                           CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SymbolVector LeakedStreams;
  StreamMapTy TrackedStreams = State->get<StreamMap>();
  for (StreamMapTy::iterator I = TrackedStreams.begin(),
                             E = TrackedStreams.end(); I != E; ++I) {
    SymbolRef Sym = I->first;
    bool IsSymDead = SymReaper.isDead(Sym);

    // Collect leaked symbols.
    if (isLeaked(Sym, I->second, IsSymDead, State))
      LeakedStreams.push_back(Sym);

    // Remove the dead symbol from the streams map.
    if (IsSymDead)
      State = State->remove<StreamMap>(Sym);
  }

  ExplodedNode *N = C.addTransition(State);
  reportLeaks(LeakedStreams, C, N);
}

void SimpleStreamChecker::reportDoubleClose(SymbolRef FileDescSym,
                                            const CallEvent &Call,
                                            CheckerContext &C) const {
  // We reached a bug, stop exploring the path here by generating a sink.
  ExplodedNode *ErrNode = C.generateSink();
  // If we've already reached this node on another path, return.
  if (!ErrNode)
    return;

  // Generate the report.
  BugReport *R = new BugReport(*DoubleCloseBugType,
      "Closing a previously closed file stream", ErrNode);
  R->addRange(Call.getSourceRange());
  R->markInteresting(FileDescSym);
  C.emitReport(R);
}

void SimpleStreamChecker::reportLeaks(SymbolVector LeakedStreams,
                                               CheckerContext &C,
                                               ExplodedNode *ErrNode) const {
  // Attach bug reports to the leak node.
  // TODO: Identify the leaked file descriptor.
  for (llvm::SmallVector<SymbolRef, 2>::iterator
      I = LeakedStreams.begin(), E = LeakedStreams.end(); I != E; ++I) {
    BugReport *R = new BugReport(*LeakBugType,
        "Opened file is never closed; potential resource leak", ErrNode);
    R->markInteresting(*I);
    C.emitReport(R);
  }
}

// Check various ways a symbol can be invalidated.
// Stop tracking symbols when a value escapes as a result of checkBind.
// A value escapes in three possible cases:
// (1) We are binding to something that is not a memory region.
// (2) We are binding to a MemRegion that does not have stack storage
// (3) We are binding to a MemRegion with stack storage that the store
//     does not understand.
void SimpleStreamChecker::checkBind(SVal loc, SVal val, const Stmt *S,
                                    CheckerContext &C) const {
  // Are we storing to something that causes the value to "escape"?
  bool escapes = true;
  ProgramStateRef state = C.getState();

  if (loc::MemRegionVal *regionLoc = dyn_cast<loc::MemRegionVal>(&loc)) {
    escapes = !regionLoc->getRegion()->hasStackStorage();

    if (!escapes) {
      // To test (3), generate a new state with the binding added.  If it is
      // the same state, then it escapes (since the store cannot represent
      // the binding). Do this only if we know that the store is not supposed
      // to generate the same state.
      SVal StoredVal = state->getSVal(regionLoc->getRegion());
      if (StoredVal != val)
        escapes = (state == (state->bindLoc(*regionLoc, val)));
    }
  }

  // If our store can represent the binding and we aren't storing to something
  // that doesn't have local storage then just return the state and
  // continue as is.
  if (!escapes)
    return;

  // Otherwise, find all symbols referenced by 'val' that we are tracking
  // and stop tracking them.
  state = state->scanReachableSymbols<StopTrackingCallback>(val).getState();
  C.addTransition(state);
}

bool SimpleStreamChecker::guaranteedNotToCloseFile(const CallEvent &Call) const{
  // If it's not in a system header, assume it might close a file.
  if (!Call.isInSystemHeader())
    return false;

  // Handle cases where we know a buffer's /address/ can escape.
  if (Call.argumentsMayEscape())
    return false;

  // Note, even though fclose closes the file, we do not list it here
  // since the checker is modeling the call.

  return true;
}

// If the symbol we are tracking is invalidated, do not track the symbol as
// we cannot reason about it anymore.
ProgramStateRef
SimpleStreamChecker::checkRegionChanges(ProgramStateRef State,
    const StoreManager::InvalidatedSymbols *invalidated,
    ArrayRef<const MemRegion *> ExplicitRegions,
    ArrayRef<const MemRegion *> Regions,
    const CallEvent *Call) const {

  if (!invalidated || invalidated->empty())
    return State;

  // If it's a call which might close the file, we assume that all regions
  // (explicit and implicit) escaped. Otherwise, whitelist explicit pointers
  // (the parameters to the call); we still can track them.
  llvm::SmallPtrSet<SymbolRef, 8> WhitelistedSymbols;
  if (!Call || guaranteedNotToCloseFile(*Call)) {
    for (ArrayRef<const MemRegion *>::iterator I = ExplicitRegions.begin(),
        E = ExplicitRegions.end(); I != E; ++I) {
      if (const SymbolicRegion *R = (*I)->StripCasts()->getAs<SymbolicRegion>())
        WhitelistedSymbols.insert(R->getSymbol());
    }
  }

  for (StoreManager::InvalidatedSymbols::const_iterator I=invalidated->begin(),
       E = invalidated->end(); I!=E; ++I) {
    SymbolRef sym = *I;
    if (WhitelistedSymbols.count(sym))
      continue;
    // The symbol escaped. Optimistically, assume that the corresponding file
    // handle will be closed somewhere else.
    State = State->remove<StreamMap>(sym);
  }
  return State;
}

void SimpleStreamChecker::initIdentifierInfo(ASTContext &Ctx) const {
  if (IIfopen)
    return;
  IIfopen = &Ctx.Idents.get("fopen");
  IIfclose = &Ctx.Idents.get("fclose");
}

void ento::registerSimpleStreamChecker(CheckerManager &mgr) {
  mgr.registerChecker<SimpleStreamChecker>();
}
