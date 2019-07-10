//===- CastValueChecker - Model implementation of custom RTTIs --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines CastValueChecker which models casts of custom RTTIs.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/ADT/Optional.h"

using namespace clang;
using namespace ento;

namespace {
class CastValueChecker : public Checker<eval::Call> {
  using CastCheck =
      std::function<void(const CastValueChecker *, const CallExpr *,
                         DefinedOrUnknownSVal, CheckerContext &)>;

public:
  // We have three cases to evaluate a cast:
  // 1) The parameter is non-null, the return value is non-null
  // 2) The parameter is non-null, the return value is null
  // 3) The parameter is null, the return value is null
  //
  // cast: 1;  dyn_cast: 1, 2;  cast_or_null: 1, 3;  dyn_cast_or_null: 1, 2, 3.
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;

private:
  // These are known in the LLVM project.
  const CallDescriptionMap<CastCheck> CDM = {
      {{{"llvm", "cast"}, 1}, &CastValueChecker::evalCast},
      {{{"llvm", "dyn_cast"}, 1}, &CastValueChecker::evalDynCast},
      {{{"llvm", "cast_or_null"}, 1}, &CastValueChecker::evalCastOrNull},
      {{{"llvm", "dyn_cast_or_null"}, 1},
       &CastValueChecker::evalDynCastOrNull}};

  void evalCast(const CallExpr *CE, DefinedOrUnknownSVal ParamDV,
                CheckerContext &C) const;
  void evalDynCast(const CallExpr *CE, DefinedOrUnknownSVal ParamDV,
                   CheckerContext &C) const;
  void evalCastOrNull(const CallExpr *CE, DefinedOrUnknownSVal ParamDV,
                      CheckerContext &C) const;
  void evalDynCastOrNull(const CallExpr *CE, DefinedOrUnknownSVal ParamDV,
                         CheckerContext &C) const;
};
} // namespace

static std::string getCastName(const Expr *Cast) {
  return Cast->getType()->getPointeeCXXRecordDecl()->getNameAsString();
}

static void evalNonNullParamNonNullReturn(const CallExpr *CE,
                                          DefinedOrUnknownSVal ParamDV,
                                          CheckerContext &C) {
  ProgramStateRef State = C.getState()->assume(ParamDV, true);
  if (!State)
    return;

  State = State->BindExpr(CE, C.getLocationContext(), ParamDV, false);

  std::string CastFromName = getCastName(CE->getArg(0));
  std::string CastToName = getCastName(CE);

  const NoteTag *CastTag = C.getNoteTag(
      [CastFromName, CastToName](BugReport &) -> std::string {
        SmallString<128> Msg;
        llvm::raw_svector_ostream Out(Msg);

        Out << "Assuming dynamic cast from '" << CastFromName << "' to '"
            << CastToName << "' succeeds";
        return Out.str();
      },
      /*IsPrunable=*/true);

  C.addTransition(State, CastTag);
}

static void evalNonNullParamNullReturn(const CallExpr *CE,
                                       DefinedOrUnknownSVal ParamDV,
                                       CheckerContext &C) {
  ProgramStateRef State = C.getState()->assume(ParamDV, true);
  if (!State)
    return;

  State = State->BindExpr(CE, C.getLocationContext(),
                          C.getSValBuilder().makeNull(), false);

  std::string CastFromName = getCastName(CE->getArg(0));
  std::string CastToName = getCastName(CE);

  const NoteTag *CastTag = C.getNoteTag(
      [CastFromName, CastToName](BugReport &) -> std::string {
        SmallString<128> Msg;
        llvm::raw_svector_ostream Out(Msg);

        Out << "Assuming dynamic cast from '" << CastFromName << "' to '"
            << CastToName << "' fails";
        return Out.str();
      },
      /*IsPrunable=*/true);

  C.addTransition(State, CastTag);
}

static void evalNullParamNullReturn(const CallExpr *CE,
                                    DefinedOrUnknownSVal ParamDV,
                                    CheckerContext &C) {
  ProgramStateRef State = C.getState()->assume(ParamDV, false);
  if (!State)
    return;

  State = State->BindExpr(CE, C.getLocationContext(),
                          C.getSValBuilder().makeNull(), false);

  const NoteTag *CastTag =
      C.getNoteTag("Assuming null pointer is passed into cast",
                   /*IsPrunable=*/true);

  C.addTransition(State, CastTag);
}

void CastValueChecker::evalCast(const CallExpr *CE,
                                DefinedOrUnknownSVal ParamDV,
                                CheckerContext &C) const {
  evalNonNullParamNonNullReturn(CE, ParamDV, C);
}

void CastValueChecker::evalDynCast(const CallExpr *CE,
                                   DefinedOrUnknownSVal ParamDV,
                                   CheckerContext &C) const {
  evalNonNullParamNonNullReturn(CE, ParamDV, C);
  evalNonNullParamNullReturn(CE, ParamDV, C);
}

void CastValueChecker::evalCastOrNull(const CallExpr *CE,
                                      DefinedOrUnknownSVal ParamDV,
                                      CheckerContext &C) const {
  evalNonNullParamNonNullReturn(CE, ParamDV, C);
  evalNullParamNullReturn(CE, ParamDV, C);
}

void CastValueChecker::evalDynCastOrNull(const CallExpr *CE,
                                         DefinedOrUnknownSVal ParamDV,
                                         CheckerContext &C) const {
  evalNonNullParamNonNullReturn(CE, ParamDV, C);
  evalNonNullParamNullReturn(CE, ParamDV, C);
  evalNullParamNullReturn(CE, ParamDV, C);
}

bool CastValueChecker::evalCall(const CallEvent &Call,
                                CheckerContext &C) const {
  const CastCheck *Check = CDM.lookup(Call);
  if (!Check)
    return false;

  const auto *CE = cast<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return false;

  // If we cannot obtain both of the classes we cannot be sure how to model it.
  if (!CE->getType()->getPointeeCXXRecordDecl() ||
      !CE->getArg(0)->getType()->getPointeeCXXRecordDecl())
    return false;

  SVal ParamV = Call.getArgSVal(0);
  auto ParamDV = ParamV.getAs<DefinedOrUnknownSVal>();
  if (!ParamDV)
    return false;

  (*Check)(this, CE, *ParamDV, C);
  return true;
}

void ento::registerCastValueChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<CastValueChecker>();
}

bool ento::shouldRegisterCastValueChecker(const LangOptions &LO) {
  return true;
}
