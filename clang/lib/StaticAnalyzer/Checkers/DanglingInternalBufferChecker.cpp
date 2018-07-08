//=== DanglingInternalBufferChecker.cpp ---------------------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a check that marks a raw pointer to a C++ container's
// inner buffer released when the object is destroyed. This information can
// be used by MallocChecker to detect use-after-free problems.
//
//===----------------------------------------------------------------------===//

#include "AllocationState.h"
#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/CommonBugCategories.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

// FIXME: member functions that return a pointer to the container's internal
// buffer may be called on the object many times, so the object's memory
// region should have a list of pointer symbols associated with it.
REGISTER_MAP_WITH_PROGRAMSTATE(RawPtrMap, const MemRegion *, SymbolRef)

namespace {

class DanglingInternalBufferChecker
    : public Checker<check::DeadSymbols, check::PostCall> {
  CallDescription CStrFn, DataFn;

public:
  class DanglingBufferBRVisitor : public BugReporterVisitor {
    SymbolRef PtrToBuf;

  public:
    DanglingBufferBRVisitor(SymbolRef Sym) : PtrToBuf(Sym) {}

    static void *getTag() {
      static int Tag = 0;
      return &Tag;
    }

    void Profile(llvm::FoldingSetNodeID &ID) const override {
      ID.AddPointer(getTag());
    }

    std::shared_ptr<PathDiagnosticPiece> VisitNode(const ExplodedNode *N,
                                                   const ExplodedNode *PrevN,
                                                   BugReporterContext &BRC,
                                                   BugReport &BR) override;

    // FIXME: Scan the map once in the visitor's constructor and do a direct
    // lookup by region.
    bool isSymbolTracked(ProgramStateRef State, SymbolRef Sym) {
      RawPtrMapTy Map = State->get<RawPtrMap>();
      for (const auto Entry : Map) {
        if (Entry.second == Sym)
          return true;
      }
      return false;
    }
  };

  DanglingInternalBufferChecker() : CStrFn("c_str"), DataFn("data") {}

  /// Record the connection between the symbol returned by c_str() and the
  /// corresponding string object region in the ProgramState. Mark the symbol
  /// released if the string object is destroyed.
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;

  /// Clean up the ProgramState map.
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
};

} // end anonymous namespace

void DanglingInternalBufferChecker::checkPostCall(const CallEvent &Call,
                                                  CheckerContext &C) const {
  const auto *ICall = dyn_cast<CXXInstanceCall>(&Call);
  if (!ICall)
    return;

  SVal Obj = ICall->getCXXThisVal();
  const auto *TypedR = dyn_cast_or_null<TypedValueRegion>(Obj.getAsRegion());
  if (!TypedR)
    return;

  auto *TypeDecl = TypedR->getValueType()->getAsCXXRecordDecl();
  if (TypeDecl->getName() != "basic_string")
    return;

  ProgramStateRef State = C.getState();

  if (Call.isCalled(CStrFn) || Call.isCalled(DataFn)) {
    SVal RawPtr = Call.getReturnValue();
    if (!RawPtr.isUnknown()) {
      State = State->set<RawPtrMap>(TypedR, RawPtr.getAsSymbol());
      C.addTransition(State);
    }
    return;
  }

  if (isa<CXXDestructorCall>(ICall)) {
    if (State->contains<RawPtrMap>(TypedR)) {
      const SymbolRef *StrBufferPtr = State->get<RawPtrMap>(TypedR);
      // FIXME: What if Origin is null?
      const Expr *Origin = Call.getOriginExpr();
      State = allocation_state::markReleased(State, *StrBufferPtr, Origin);
      State = State->remove<RawPtrMap>(TypedR);
      C.addTransition(State);
      return;
    }
  }
}

void DanglingInternalBufferChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                                     CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  RawPtrMapTy RPM = State->get<RawPtrMap>();
  for (const auto Entry : RPM) {
    if (!SymReaper.isLive(Entry.second))
      State = State->remove<RawPtrMap>(Entry.first);
    if (!SymReaper.isLiveRegion(Entry.first)) {
      // Due to incomplete destructor support, some dead regions might still
      // remain in the program state map. Clean them up.
      State = State->remove<RawPtrMap>(Entry.first);
    }
  }
  C.addTransition(State);
}

std::shared_ptr<PathDiagnosticPiece>
DanglingInternalBufferChecker::DanglingBufferBRVisitor::VisitNode(
    const ExplodedNode *N, const ExplodedNode *PrevN, BugReporterContext &BRC,
    BugReport &BR) {

  if (!isSymbolTracked(N->getState(), PtrToBuf) ||
      isSymbolTracked(PrevN->getState(), PtrToBuf))
    return nullptr;

  const Stmt *S = PathDiagnosticLocation::getStmt(N);
  if (!S)
    return nullptr;

  SmallString<256> Buf;
  llvm::raw_svector_ostream OS(Buf);
  OS << "Pointer to dangling buffer was obtained here";
  PathDiagnosticLocation Pos(S, BRC.getSourceManager(),
                             N->getLocationContext());
  return std::make_shared<PathDiagnosticEventPiece>(Pos, OS.str(), true,
                                                    nullptr);
}

namespace clang {
namespace ento {
namespace allocation_state {

std::unique_ptr<BugReporterVisitor> getDanglingBufferBRVisitor(SymbolRef Sym) {
  return llvm::make_unique<
      DanglingInternalBufferChecker::DanglingBufferBRVisitor>(Sym);
}

} // end namespace allocation_state
} // end namespace ento
} // end namespace clang

void ento::registerDanglingInternalBufferChecker(CheckerManager &Mgr) {
  registerNewDeleteChecker(Mgr);
  Mgr.registerChecker<DanglingInternalBufferChecker>();
}
