//== InvalidPtrChecker.cpp ------------------------------------- -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines InvalidPtrChecker which finds usages of possibly
// invalidated pointer.
// CERT SEI Rules ENV31-C and ENV34-C
// For more information see:
// https://wiki.sei.cmu.edu/confluence/x/8tYxBQ
// https://wiki.sei.cmu.edu/confluence/x/5NUxBQ
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

class InvalidPtrChecker
    : public Checker<check::Location, check::BeginFunction, check::PostCall> {
private:
  BugType BT{this, "Use of invalidated pointer", categories::MemoryError};

  void EnvpInvalidatingCall(const CallEvent &Call, CheckerContext &C) const;

  using HandlerFn = void (InvalidPtrChecker::*)(const CallEvent &Call,
                                                CheckerContext &C) const;

  // SEI CERT ENV31-C
  const CallDescriptionMap<HandlerFn> EnvpInvalidatingFunctions = {
      {{"setenv", 3}, &InvalidPtrChecker::EnvpInvalidatingCall},
      {{"unsetenv", 1}, &InvalidPtrChecker::EnvpInvalidatingCall},
      {{"putenv", 1}, &InvalidPtrChecker::EnvpInvalidatingCall},
      {{"_putenv_s", 2}, &InvalidPtrChecker::EnvpInvalidatingCall},
      {{"_wputenv_s", 2}, &InvalidPtrChecker::EnvpInvalidatingCall},
  };

  void postPreviousReturnInvalidatingCall(const CallEvent &Call,
                                          CheckerContext &C) const;

  // SEI CERT ENV34-C
  const CallDescriptionMap<HandlerFn> PreviousCallInvalidatingFunctions = {
      {{"getenv", 1}, &InvalidPtrChecker::postPreviousReturnInvalidatingCall},
      {{"setlocale", 2},
       &InvalidPtrChecker::postPreviousReturnInvalidatingCall},
      {{"strerror", 1}, &InvalidPtrChecker::postPreviousReturnInvalidatingCall},
      {{"localeconv", 0},
       &InvalidPtrChecker::postPreviousReturnInvalidatingCall},
      {{"asctime", 1}, &InvalidPtrChecker::postPreviousReturnInvalidatingCall},
  };

public:
  // Obtain the environment pointer from 'main()' (if present).
  void checkBeginFunction(CheckerContext &C) const;

  // Handle functions in EnvpInvalidatingFunctions, that invalidate environment
  // pointer from 'main()'
  // Handle functions in PreviousCallInvalidatingFunctions.
  // Also, check if invalidated region is passed to a
  // conservatively evaluated function call as an argument.
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;

  // Check if invalidated region is being dereferenced.
  void checkLocation(SVal l, bool isLoad, const Stmt *S,
                     CheckerContext &C) const;
};

} // namespace

// Set of memory regions that were invalidated
REGISTER_SET_WITH_PROGRAMSTATE(InvalidMemoryRegions, const MemRegion *)

// Stores the region of the environment pointer of 'main' (if present).
// Note: This pointer has type 'const MemRegion *', however the trait is only
// specialized to 'const void*' and 'void*'
REGISTER_TRAIT_WITH_PROGRAMSTATE(EnvPtrRegion, const void *)

// Stores key-value pairs, where key is function declaration and value is
// pointer to memory region returned by previous call of this function
REGISTER_MAP_WITH_PROGRAMSTATE(PreviousCallResultMap, const FunctionDecl *,
                               const MemRegion *)

void InvalidPtrChecker::EnvpInvalidatingCall(const CallEvent &Call,
                                             CheckerContext &C) const {
  StringRef FunctionName = Call.getCalleeIdentifier()->getName();
  ProgramStateRef State = C.getState();
  const auto *Reg = State->get<EnvPtrRegion>();
  if (!Reg)
    return;
  const auto *SymbolicEnvPtrRegion =
      reinterpret_cast<const MemRegion *>(const_cast<const void *>(Reg));

  State = State->add<InvalidMemoryRegions>(SymbolicEnvPtrRegion);

  const NoteTag *Note =
      C.getNoteTag([SymbolicEnvPtrRegion, FunctionName](
                       PathSensitiveBugReport &BR, llvm::raw_ostream &Out) {
        if (!BR.isInteresting(SymbolicEnvPtrRegion))
          return;
        Out << '\'' << FunctionName
            << "' call may invalidate the environment parameter of 'main'";
      });

  C.addTransition(State, Note);
}

void InvalidPtrChecker::postPreviousReturnInvalidatingCall(
    const CallEvent &Call, CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  const NoteTag *Note = nullptr;
  const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  // Invalidate the region of the previously returned pointer - if there was
  // one.
  if (const MemRegion *const *Reg = State->get<PreviousCallResultMap>(FD)) {
    const MemRegion *PrevReg = *Reg;
    State = State->add<InvalidMemoryRegions>(PrevReg);
    Note = C.getNoteTag([PrevReg, FD](PathSensitiveBugReport &BR,
                                      llvm::raw_ostream &Out) {
      if (!BR.isInteresting(PrevReg))
        return;
      Out << '\'';
      FD->getNameForDiagnostic(Out, FD->getASTContext().getLangOpts(), true);
      Out << "' call may invalidate the result of the previous " << '\'';
      FD->getNameForDiagnostic(Out, FD->getASTContext().getLangOpts(), true);
      Out << '\'';
    });
  }

  const LocationContext *LCtx = C.getLocationContext();
  const auto *CE = cast<CallExpr>(Call.getOriginExpr());

  // Function call will return a pointer to the new symbolic region.
  DefinedOrUnknownSVal RetVal = C.getSValBuilder().conjureSymbolVal(
      CE, LCtx, CE->getType(), C.blockCount());
  State = State->BindExpr(CE, LCtx, RetVal);

  // Remember to this region.
  const auto *SymRegOfRetVal = cast<SymbolicRegion>(RetVal.getAsRegion());
  const MemRegion *MR =
      const_cast<MemRegion *>(SymRegOfRetVal->getBaseRegion());
  State = State->set<PreviousCallResultMap>(FD, MR);

  ExplodedNode *Node = C.addTransition(State, Note);
  const NoteTag *PreviousCallNote =
      C.getNoteTag([MR](PathSensitiveBugReport &BR, llvm::raw_ostream &Out) {
        if (!BR.isInteresting(MR))
          return;
        Out << '\'' << "'previous function call was here" << '\'';
      });

  C.addTransition(State, Node, PreviousCallNote);
}

// TODO: This seems really ugly. Simplify this.
static const MemRegion *findInvalidatedSymbolicBase(ProgramStateRef State,
                                                    const MemRegion *Reg) {
  while (Reg) {
    if (State->contains<InvalidMemoryRegions>(Reg))
      return Reg;
    const auto *SymBase = Reg->getSymbolicBase();
    if (!SymBase)
      break;
    const auto *SRV = dyn_cast<SymbolRegionValue>(SymBase->getSymbol());
    if (!SRV)
      break;
    Reg = SRV->getRegion();
    if (const auto *VarReg = dyn_cast<VarRegion>(SRV->getRegion()))
      Reg = VarReg;
  }
  return nullptr;
}

// Handle functions in EnvpInvalidatingFunctions, that invalidate environment
// pointer from 'main()' Also, check if invalidated region is passed to a
// function call as an argument.
void InvalidPtrChecker::checkPostCall(const CallEvent &Call,
                                      CheckerContext &C) const {
  // Check if function invalidates 'envp' argument of 'main'
  if (const auto *Handler = EnvpInvalidatingFunctions.lookup(Call))
    (this->**Handler)(Call, C);

  // Check if function invalidates the result of previous call
  if (const auto *Handler = PreviousCallInvalidatingFunctions.lookup(Call))
    (this->**Handler)(Call, C);

  // Check if one of the arguments of the function call is invalidated

  // If call was inlined, don't report invalidated argument
  if (C.wasInlined)
    return;

  ProgramStateRef State = C.getState();

  for (unsigned I = 0, NumArgs = Call.getNumArgs(); I < NumArgs; ++I) {

    if (const auto *SR = dyn_cast_or_null<SymbolicRegion>(
            Call.getArgSVal(I).getAsRegion())) {
      if (const MemRegion *InvalidatedSymbolicBase =
              findInvalidatedSymbolicBase(State, SR)) {
        ExplodedNode *ErrorNode = C.generateNonFatalErrorNode();
        if (!ErrorNode)
          return;

        SmallString<256> Msg;
        llvm::raw_svector_ostream Out(Msg);
        Out << "use of invalidated pointer '";
        Call.getArgExpr(I)->printPretty(Out, /*Helper=*/nullptr,
                                        C.getASTContext().getPrintingPolicy());
        Out << "' in a function call";

        auto Report =
            std::make_unique<PathSensitiveBugReport>(BT, Out.str(), ErrorNode);
        Report->markInteresting(InvalidatedSymbolicBase);
        Report->addRange(Call.getArgSourceRange(I));
        C.emitReport(std::move(Report));
      }
    }
  }
}

// Obtain the environment pointer from 'main()', if present.
void InvalidPtrChecker::checkBeginFunction(CheckerContext &C) const {
  if (!C.inTopFrame())
    return;

  const auto *FD = dyn_cast<FunctionDecl>(C.getLocationContext()->getDecl());
  if (!FD || FD->param_size() != 3 || !FD->isMain())
    return;

  ProgramStateRef State = C.getState();
  const MemRegion *EnvpReg =
      State->getRegion(FD->parameters()[2], C.getLocationContext());

  // Save the memory region pointed by the environment pointer parameter of
  // 'main'.
  State = State->set<EnvPtrRegion>(
      reinterpret_cast<void *>(const_cast<MemRegion *>(EnvpReg)));
  C.addTransition(State);
}

// Check if invalidated region is being dereferenced.
void InvalidPtrChecker::checkLocation(SVal Loc, bool isLoad, const Stmt *S,
                                      CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  // Ignore memory operations involving 'non-invalidated' locations.
  const MemRegion *InvalidatedSymbolicBase =
      findInvalidatedSymbolicBase(State, Loc.getAsRegion());
  if (!InvalidatedSymbolicBase)
    return;

  ExplodedNode *ErrorNode = C.generateNonFatalErrorNode();
  if (!ErrorNode)
    return;

  auto Report = std::make_unique<PathSensitiveBugReport>(
      BT, "dereferencing an invalid pointer", ErrorNode);
  Report->markInteresting(InvalidatedSymbolicBase);
  C.emitReport(std::move(Report));
}

void ento::registerInvalidPtrChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<InvalidPtrChecker>();
}

bool ento::shouldRegisterInvalidPtrChecker(const CheckerManager &) {
  return true;
}
