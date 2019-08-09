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
#include <utility>

using namespace clang;
using namespace ento;

namespace {
class CastValueChecker : public Checker<eval::Call> {
  enum class CastKind { Function, Method };

  using CastCheck =
      std::function<void(const CastValueChecker *, const CallExpr *,
                         DefinedOrUnknownSVal, CheckerContext &)>;

  using CheckKindPair = std::pair<CastCheck, CastKind>;

public:
  // We have five cases to evaluate a cast:
  // 1) The parameter is non-null, the return value is non-null
  // 2) The parameter is non-null, the return value is null
  // 3) The parameter is null, the return value is null
  // cast: 1;  dyn_cast: 1, 2;  cast_or_null: 1, 3;  dyn_cast_or_null: 1, 2, 3.
  //
  // 4) castAs: has no parameter, the return value is non-null.
  // 5) getAs:  has no parameter, the return value is null or non-null.
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;

private:
  // These are known in the LLVM project. The pairs are in the following form:
  // {{{namespace, call}, argument-count}, {callback, kind}}
  const CallDescriptionMap<CheckKindPair> CDM = {
      {{{"llvm", "cast"}, 1},
       {&CastValueChecker::evalCast, CastKind::Function}},
      {{{"llvm", "dyn_cast"}, 1},
       {&CastValueChecker::evalDynCast, CastKind::Function}},
      {{{"llvm", "cast_or_null"}, 1},
       {&CastValueChecker::evalCastOrNull, CastKind::Function}},
      {{{"llvm", "dyn_cast_or_null"}, 1},
       {&CastValueChecker::evalDynCastOrNull, CastKind::Function}},
      {{{"clang", "castAs"}, 0},
       {&CastValueChecker::evalCastAs, CastKind::Method}},
      {{{"clang", "getAs"}, 0},
       {&CastValueChecker::evalGetAs, CastKind::Method}}};

  void evalCast(const CallExpr *CE, DefinedOrUnknownSVal DV,
                CheckerContext &C) const;
  void evalDynCast(const CallExpr *CE, DefinedOrUnknownSVal DV,
                   CheckerContext &C) const;
  void evalCastOrNull(const CallExpr *CE, DefinedOrUnknownSVal DV,
                      CheckerContext &C) const;
  void evalDynCastOrNull(const CallExpr *CE, DefinedOrUnknownSVal DV,
                         CheckerContext &C) const;
  void evalCastAs(const CallExpr *CE, DefinedOrUnknownSVal DV,
                  CheckerContext &C) const;
  void evalGetAs(const CallExpr *CE, DefinedOrUnknownSVal DV,
                 CheckerContext &C) const;
};
} // namespace

static std::string getCastName(const Expr *Cast) {
  QualType Ty = Cast->getType();
  if (const CXXRecordDecl *RD = Ty->getAsCXXRecordDecl())
    return RD->getNameAsString();

  return Ty->getPointeeCXXRecordDecl()->getNameAsString();
}

static const NoteTag *getCastTag(bool IsNullReturn, const CallExpr *CE,
                                 CheckerContext &C,
                                 bool IsCheckedCast = false) {
  Optional<std::string> CastFromName = (CE->getNumArgs() > 0)
                                           ? getCastName(CE->getArg(0))
                                           : Optional<std::string>();
  std::string CastToName = getCastName(CE);

  return C.getNoteTag(
      [CastFromName, CastToName, IsNullReturn,
       IsCheckedCast](BugReport &) -> std::string {
        SmallString<128> Msg;
        llvm::raw_svector_ostream Out(Msg);

        Out << (!IsCheckedCast ? "Assuming dynamic cast " : "Checked cast ");
        if (CastFromName)
          Out << "from '" << *CastFromName << "' ";

        Out << "to '" << CastToName << "' "
            << (!IsNullReturn ? "succeeds" : "fails");

        return Out.str();
      },
      /*IsPrunable=*/true);
}

static ProgramStateRef getState(bool IsNullReturn,
                                DefinedOrUnknownSVal ReturnDV,
                                const CallExpr *CE, ProgramStateRef State,
                                CheckerContext &C) {
  return State->BindExpr(
      CE, C.getLocationContext(),
      IsNullReturn ? C.getSValBuilder().makeNull() : ReturnDV, false);
}

//===----------------------------------------------------------------------===//
// Evaluating cast, dyn_cast, cast_or_null, dyn_cast_or_null.
//===----------------------------------------------------------------------===//

static void evalNonNullParamNonNullReturn(const CallExpr *CE,
                                          DefinedOrUnknownSVal DV,
                                          CheckerContext &C,
                                          bool IsCheckedCast = false) {
  bool IsNullReturn = false;
  if (ProgramStateRef State = C.getState()->assume(DV, true))
    C.addTransition(getState(IsNullReturn, DV, CE, State, C),
                    getCastTag(IsNullReturn, CE, C, IsCheckedCast));
}

static void evalNonNullParamNullReturn(const CallExpr *CE,
                                       DefinedOrUnknownSVal DV,
                                       CheckerContext &C) {
  bool IsNullReturn = true;
  if (ProgramStateRef State = C.getState()->assume(DV, true))
    C.addTransition(getState(IsNullReturn, DV, CE, State, C),
                    getCastTag(IsNullReturn, CE, C));
}

static void evalNullParamNullReturn(const CallExpr *CE, DefinedOrUnknownSVal DV,
                                    CheckerContext &C) {
  if (ProgramStateRef State = C.getState()->assume(DV, false))
    C.addTransition(getState(/*IsNullReturn=*/true, DV, CE, State, C),
                    C.getNoteTag("Assuming null pointer is passed into cast",
                                 /*IsPrunable=*/true));
}

void CastValueChecker::evalCast(const CallExpr *CE, DefinedOrUnknownSVal DV,
                                CheckerContext &C) const {
  evalNonNullParamNonNullReturn(CE, DV, C, /*IsCheckedCast=*/true);
}

void CastValueChecker::evalDynCast(const CallExpr *CE, DefinedOrUnknownSVal DV,
                                   CheckerContext &C) const {
  evalNonNullParamNonNullReturn(CE, DV, C);
  evalNonNullParamNullReturn(CE, DV, C);
}

void CastValueChecker::evalCastOrNull(const CallExpr *CE,
                                      DefinedOrUnknownSVal DV,
                                      CheckerContext &C) const {
  evalNonNullParamNonNullReturn(CE, DV, C);
  evalNullParamNullReturn(CE, DV, C);
}

void CastValueChecker::evalDynCastOrNull(const CallExpr *CE,
                                         DefinedOrUnknownSVal DV,
                                         CheckerContext &C) const {
  evalNonNullParamNonNullReturn(CE, DV, C);
  evalNonNullParamNullReturn(CE, DV, C);
  evalNullParamNullReturn(CE, DV, C);
}

//===----------------------------------------------------------------------===//
// Evaluating castAs, getAs.
//===----------------------------------------------------------------------===//

static void evalZeroParamNonNullReturn(const CallExpr *CE,
                                       DefinedOrUnknownSVal DV,
                                       CheckerContext &C,
                                       bool IsCheckedCast = false) {
  bool IsNullReturn = false;
  if (ProgramStateRef State = C.getState()->assume(DV, true))
    C.addTransition(getState(IsNullReturn, DV, CE, C.getState(), C),
                    getCastTag(IsNullReturn, CE, C, IsCheckedCast));
}

static void evalZeroParamNullReturn(const CallExpr *CE, DefinedOrUnknownSVal DV,
                                    CheckerContext &C) {
  bool IsNullReturn = true;
  if (ProgramStateRef State = C.getState()->assume(DV, true))
    C.addTransition(getState(IsNullReturn, DV, CE, C.getState(), C),
                    getCastTag(IsNullReturn, CE, C));
}

void CastValueChecker::evalCastAs(const CallExpr *CE, DefinedOrUnknownSVal DV,
                                  CheckerContext &C) const {
  evalZeroParamNonNullReturn(CE, DV, C, /*IsCheckedCast=*/true);
}

void CastValueChecker::evalGetAs(const CallExpr *CE, DefinedOrUnknownSVal DV,
                                 CheckerContext &C) const {
  evalZeroParamNonNullReturn(CE, DV, C);
  evalZeroParamNullReturn(CE, DV, C);
}

bool CastValueChecker::evalCall(const CallEvent &Call,
                                CheckerContext &C) const {
  const auto *Lookup = CDM.lookup(Call);
  if (!Lookup)
    return false;

  // If we cannot obtain the call's class we cannot be sure how to model it.
  QualType ResultTy = Call.getResultType();
  if (!ResultTy->getPointeeCXXRecordDecl())
    return false;

  const CastCheck &Check = Lookup->first;
  CastKind Kind = Lookup->second;

  const auto *CE = cast<CallExpr>(Call.getOriginExpr());
  Optional<DefinedOrUnknownSVal> DV;

  switch (Kind) {
  case CastKind::Function: {
    // If we cannot obtain the arg's class we cannot be sure how to model it.
    QualType ArgTy = Call.parameters()[0]->getType();
    if (!ArgTy->getAsCXXRecordDecl() && !ArgTy->getPointeeCXXRecordDecl())
      return false;

    DV = Call.getArgSVal(0).getAs<DefinedOrUnknownSVal>();
    break;
  }
  case CastKind::Method:
    // If we cannot obtain the 'InstanceCall' we cannot be sure how to model it.
    const auto *InstanceCall = dyn_cast<CXXInstanceCall>(&Call);
    if (!InstanceCall)
      return false;

    DV = InstanceCall->getCXXThisVal().getAs<DefinedOrUnknownSVal>();
    break;
  }

  if (!DV)
    return false;

  Check(this, CE, *DV, C);
  return true;
}

void ento::registerCastValueChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<CastValueChecker>();
}

bool ento::shouldRegisterCastValueChecker(const LangOptions &LO) {
  return true;
}
