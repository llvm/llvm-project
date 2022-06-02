//=== ErrnoModeling.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines a checker `ErrnoModeling`, which is used to make the system
// value 'errno' available to other checkers.
// The 'errno' value is stored at a special memory region that is accessible
// through the `errno_modeling` namespace. The memory region is either the
// region of `errno` itself if it is a variable, otherwise an artifically
// created region (in the system memory space). If `errno` is defined by using
// a function which returns the address of it (this is always the case if it is
// not a variable) this function is recognized and evaluated. In this way
// `errno` becomes visible to the analysis and checkers can change its value.
//
//===----------------------------------------------------------------------===//

#include "ErrnoModeling.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang;
using namespace ento;

namespace {

// Name of the "errno" variable.
// FIXME: Is there a system where it is not called "errno" but is a variable?
const char *ErrnoVarName = "errno";
// Names of functions that return a location of the "errno" value.
// FIXME: Are there other similar function names?
const char *ErrnoLocationFuncNames[] = {"__errno_location", "___errno",
                                        "__errno", "_errno", "__error"};

class ErrnoModeling
    : public Checker<check::ASTDecl<TranslationUnitDecl>, check::BeginFunction,
                     check::LiveSymbols, eval::Call> {
public:
  void checkASTDecl(const TranslationUnitDecl *D, AnalysisManager &Mgr,
                    BugReporter &BR) const;
  void checkBeginFunction(CheckerContext &C) const;
  void checkLiveSymbols(ProgramStateRef State, SymbolReaper &SR) const;
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;

  // The declaration of an "errno" variable or "errno location" function.
  mutable const Decl *ErrnoDecl = nullptr;

private:
  // FIXME: Names from `ErrnoLocationFuncNames` are used to build this set.
  CallDescriptionSet ErrnoLocationCalls{{"__errno_location", 0, 0},
                                        {"___errno", 0, 0},
                                        {"__errno", 0, 0},
                                        {"_errno", 0, 0},
                                        {"__error", 0, 0}};
};

} // namespace

/// Store a MemRegion that contains the 'errno' integer value.
/// The value is null if the 'errno' value was not recognized in the AST.
REGISTER_TRAIT_WITH_PROGRAMSTATE(ErrnoRegion, const MemRegion *)

/// Search for a variable called "errno" in the AST.
/// Return nullptr if not found.
static const VarDecl *getErrnoVar(ASTContext &ACtx) {
  IdentifierInfo &II = ACtx.Idents.get(ErrnoVarName);
  auto LookupRes = ACtx.getTranslationUnitDecl()->lookup(&II);
  auto Found = llvm::find_if(LookupRes, [&ACtx](const Decl *D) {
    if (auto *VD = dyn_cast<VarDecl>(D))
      return ACtx.getSourceManager().isInSystemHeader(VD->getLocation()) &&
             VD->hasExternalStorage() &&
             VD->getType().getCanonicalType() == ACtx.IntTy;
    return false;
  });
  if (Found == LookupRes.end())
    return nullptr;

  return cast<VarDecl>(*Found);
}

/// Search for a function with a specific name that is used to return a pointer
/// to "errno".
/// Return nullptr if no such function was found.
static const FunctionDecl *getErrnoFunc(ASTContext &ACtx) {
  SmallVector<const Decl *> LookupRes;
  for (StringRef ErrnoName : ErrnoLocationFuncNames) {
    IdentifierInfo &II = ACtx.Idents.get(ErrnoName);
    llvm::append_range(LookupRes, ACtx.getTranslationUnitDecl()->lookup(&II));
  }

  auto Found = llvm::find_if(LookupRes, [&ACtx](const Decl *D) {
    if (auto *FD = dyn_cast<FunctionDecl>(D))
      return ACtx.getSourceManager().isInSystemHeader(FD->getLocation()) &&
             FD->isExternC() && FD->getNumParams() == 0 &&
             FD->getReturnType().getCanonicalType() ==
                 ACtx.getPointerType(ACtx.IntTy);
    return false;
  });
  if (Found == LookupRes.end())
    return nullptr;

  return cast<FunctionDecl>(*Found);
}

void ErrnoModeling::checkASTDecl(const TranslationUnitDecl *D,
                                 AnalysisManager &Mgr, BugReporter &BR) const {
  // Try to find an usable `errno` value.
  // It can be an external variable called "errno" or a function that returns a
  // pointer to the "errno" value. This function can have different names.
  // The actual case is dependent on the C library implementation, we
  // can only search for a match in one of these variations.
  // We assume that exactly one of these cases might be true.
  ErrnoDecl = getErrnoVar(Mgr.getASTContext());
  if (!ErrnoDecl)
    ErrnoDecl = getErrnoFunc(Mgr.getASTContext());
}

void ErrnoModeling::checkBeginFunction(CheckerContext &C) const {
  if (!C.inTopFrame())
    return;

  ASTContext &ACtx = C.getASTContext();
  ProgramStateRef State = C.getState();

  if (const auto *ErrnoVar = dyn_cast_or_null<VarDecl>(ErrnoDecl)) {
    // There is an external 'errno' variable.
    // Use its memory region.
    // The memory region for an 'errno'-like variable is allocated in system
    // space by MemRegionManager.
    const MemRegion *ErrnoR =
        State->getRegion(ErrnoVar, C.getLocationContext());
    assert(ErrnoR && "Memory region should exist for the 'errno' variable.");
    State = State->set<ErrnoRegion>(ErrnoR);
    State = errno_modeling::setErrnoValue(State, C, 0);
    C.addTransition(State);
  } else if (ErrnoDecl) {
    assert(isa<FunctionDecl>(ErrnoDecl) && "Invalid errno location function.");
    // There is a function that returns the location of 'errno'.
    // We must create a memory region for it in system space.
    // Currently a symbolic region is used with an artifical symbol.
    // FIXME: It is better to have a custom (new) kind of MemRegion for such
    // cases.
    SValBuilder &SVB = C.getSValBuilder();
    MemRegionManager &RMgr = C.getStateManager().getRegionManager();

    const MemSpaceRegion *GlobalSystemSpace =
        RMgr.getGlobalsRegion(MemRegion::GlobalSystemSpaceRegionKind);

    // Create an artifical symbol for the region.
    // It is not possible to associate a statement or expression in this case.
    const SymbolConjured *Sym = SVB.conjureSymbol(
        nullptr, C.getLocationContext(),
        ACtx.getLValueReferenceType(ACtx.IntTy), C.blockCount(), &ErrnoDecl);

    // The symbolic region is untyped, create a typed sub-region in it.
    // The ElementRegion is used to make the errno region a typed region.
    const MemRegion *ErrnoR = RMgr.getElementRegion(
        ACtx.IntTy, SVB.makeZeroArrayIndex(),
        RMgr.getSymbolicRegion(Sym, GlobalSystemSpace), C.getASTContext());
    State = State->set<ErrnoRegion>(ErrnoR);
    State = errno_modeling::setErrnoValue(State, C, 0);
    C.addTransition(State);
  }
}

bool ErrnoModeling::evalCall(const CallEvent &Call, CheckerContext &C) const {
  // Return location of "errno" at a call to an "errno address returning"
  // function.
  if (ErrnoLocationCalls.contains(Call)) {
    ProgramStateRef State = C.getState();

    const MemRegion *ErrnoR = State->get<ErrnoRegion>();
    if (!ErrnoR)
      return false;

    State = State->BindExpr(Call.getOriginExpr(), C.getLocationContext(),
                            loc::MemRegionVal{ErrnoR});
    C.addTransition(State);
    return true;
  }

  return false;
}

void ErrnoModeling::checkLiveSymbols(ProgramStateRef State,
                                     SymbolReaper &SR) const {
  // The special errno region should never garbage collected.
  if (const MemRegion *ErrnoR = State->get<ErrnoRegion>())
    SR.markLive(ErrnoR);
}

namespace clang {
namespace ento {
namespace errno_modeling {

Optional<SVal> getErrnoValue(ProgramStateRef State) {
  const MemRegion *ErrnoR = State->get<ErrnoRegion>();
  if (!ErrnoR)
    return {};
  QualType IntTy = State->getAnalysisManager().getASTContext().IntTy;
  return State->getSVal(ErrnoR, IntTy);
}

ProgramStateRef setErrnoValue(ProgramStateRef State,
                              const LocationContext *LCtx, SVal Value) {
  const MemRegion *ErrnoR = State->get<ErrnoRegion>();
  if (!ErrnoR)
    return State;
  return State->bindLoc(loc::MemRegionVal{ErrnoR}, Value, LCtx);
}

ProgramStateRef setErrnoValue(ProgramStateRef State, CheckerContext &C,
                              uint64_t Value) {
  const MemRegion *ErrnoR = State->get<ErrnoRegion>();
  if (!ErrnoR)
    return State;
  return State->bindLoc(
      loc::MemRegionVal{ErrnoR},
      C.getSValBuilder().makeIntVal(Value, C.getASTContext().IntTy),
      C.getLocationContext());
}

} // namespace errno_modeling
} // namespace ento
} // namespace clang

void ento::registerErrnoModeling(CheckerManager &mgr) {
  mgr.registerChecker<ErrnoModeling>();
}

bool ento::shouldRegisterErrnoModeling(const CheckerManager &mgr) {
  return true;
}
