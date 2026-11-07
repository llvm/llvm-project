//== FsetposChecker.cpp ---------------------------------------- -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines FsetposChecker which finds 'fsetpos' calls that do not take
// a 'pos' argument returned from 'fgetpos'.
// CERT SEI Rule FIO44-C
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {

class FsetposChecker
    : public Checker<check::Location, check::PreCall, check::PostCall,
                     check::PointerEscape, check::DeadSymbols> {
private:
  BugType BT_InvalidFsetposArg{this,
                               "Call of 'fsetpos' with position argument that "
                               "was not obtained by 'fgetpos'",
                               categories::UnixAPI};

  CallDescription FgetposCall{CDM::CLibrary, {"fgetpos"}, 2};
  CallDescription FsetposCall{CDM::CLibrary, {"fsetpos"}, 2};

public:
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkLocation(SVal L, bool IsLoad, const Stmt *S,
                     CheckerContext &C) const;
  ProgramStateRef checkPointerEscape(ProgramStateRef State,
                                     const InvalidatedSymbols &Escaped,
                                     const CallEvent *Call,
                                     PointerEscapeKind Kind) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
};

} // namespace

/// Used to get the 'FILE *' value from a 'fpos_t *' value or the SVal at this
/// location.
/// For example at 'fsetpos(F, &P)' the 'F' value can be get by using '&P'
/// (pointer) or the 'P' (a value of 'fpos_t' type) as key. The 'fpos_t *' is
/// used as key because the 'fpos_t' type can be a struct, in this case it may
/// not have a permanent SVal. (But if it is an integral type tracking the
/// value works even at assignments of 'fpos_t' type.)
REGISTER_MAP_WITH_PROGRAMSTATE(FPosFile, SVal, SVal)

namespace {

SVal getFileVal(ProgramStateRef State, SVal LocVal, bool *IsInteger) {
  if (LocVal.isUndef())
    return UndefinedVal{};
  auto FPosLoc = LocVal.getAs<Loc>();
  if (!FPosLoc)
    return UnknownVal{};
  SVal FPosVal = State->getSVal(*FPosLoc);
  if (FPosVal.getAsInteger())
    *IsInteger = true;
  if (FPosVal.getAsInteger() || FPosVal.isUndef())
    return UndefinedVal{};
  if (const SVal *V = State->get<FPosFile>(*FPosLoc))
    return *V;
  if (!FPosVal.isUnknownOrUndef())
    if (const SVal *V = State->get<FPosFile>(FPosVal))
      return *V;
  return UnknownVal{};
}

} // namespace

void FsetposChecker::checkPreCall(const CallEvent &Call,
                                  CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  if (FsetposCall.matches(Call)) {
    SVal FileVal = Call.getArgSVal(0);
    if (FileVal.isUnknownOrUndef())
      return;
    SVal FPosLocVal = Call.getArgSVal(1);
    bool IsInteger = false;
    SVal FPosFileVal = getFileVal(State, FPosLocVal, &IsInteger);
    if (FPosFileVal.isUnknown())
      return;
    if (FPosFileVal.isUndef() || FPosFileVal != FileVal) {
      if (ExplodedNode *N = C.generateErrorNode()) {
        SmallString<100> Str;
        llvm::raw_svector_ostream Os(Str);
        Os << "Position argument to 'fsetpos' ";
        if (IsInteger)
          Os << "(which is a constant value) ";
        else if (FPosFileVal.isUndef())
          Os << "(which is an undefined value) ";
        Os << "was not obained from previous call to 'fgetpos' on the same "
              "file";
        auto R = std::make_unique<PathSensitiveBugReport>(BT_InvalidFsetposArg,
                                                          Str, N);
        R->addRange(Call.getArgSourceRange(1));
        C.emitReport(std::move(R));
      }
    }
  }
}

void FsetposChecker::checkPostCall(const CallEvent &Call,
                                   CheckerContext &C) const {

  ProgramStateRef State = C.getState();

  if (FgetposCall.matches(Call)) {
    SVal FileVal = Call.getArgSVal(0);
    auto FPosLoc = Call.getArgSVal(1).getAs<Loc>();
    if (!FPosLoc)
      return;
    if (FileVal.isUnknownOrUndef())
      return;

    State = State->set<FPosFile>(*FPosLoc, FileVal);
    if (SVal FPosVal = State->getSVal(*FPosLoc); !FPosVal.isUnknownOrUndef())
      State = State->set<FPosFile>(FPosVal, FileVal);

    C.addTransition(State);
    return;
  }
}

void FsetposChecker::checkLocation(SVal L, bool IsLoad, const Stmt *S,
                                   CheckerContext &C) const {
  if (IsLoad)
    return;
  ProgramStateRef State = C.getState();
  if (State->get<FPosFile>(L)) {
    State = State->remove<FPosFile>(L);
    C.addTransition(State);
  }
}

ProgramStateRef FsetposChecker::checkPointerEscape(
    ProgramStateRef State, const InvalidatedSymbols &Escaped,
    const CallEvent *Call, PointerEscapeKind Kind) const {
  FPosFileTy FPF = State->get<FPosFile>();
  for (auto [Loc, F] : FPF) {
    if (Escaped.contains(Loc.getAsSymbol()) ||
        Escaped.contains(F.getAsSymbol()))
      State = State->remove<FPosFile>(Loc);
  }
  return State;
}

void FsetposChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                      CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  FPosFileTy FPF = State->get<FPosFile>();
  for (auto [Loc, F] : FPF) {
    SymbolRef LocSym = Loc.getAsSymbol();
    SymbolRef FSym = F.getAsSymbol();
    if ((LocSym && SymReaper.isDead(LocSym)) ||
        (FSym && SymReaper.isDead(FSym)))
      State = State->remove<FPosFile>(Loc);
  }
  C.addTransition(State);
}

void ento::registerFsetposChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<FsetposChecker>();
}

bool ento::shouldRegisterFsetposChecker(const CheckerManager &) { return true; }
