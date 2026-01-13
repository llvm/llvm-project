//== Checker.h - Registration mechanism for checkers -------------*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines Checker, used to create and register checkers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_CHECKER_H
#define LLVM_CLANG_STATICANALYZER_CORE_CHECKER_H

#include "clang/Analysis/ProgramPoint.h"
#include "clang/Basic/LangOptions.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "llvm/Support/Casting.h"

namespace clang {
namespace ento {
  class BugReporter;

namespace check {

template <typename DECL>
class ASTDecl {
  template <typename CHECKER>
  static void _checkDecl(void *checker, const Decl *D, AnalysisManager& mgr,
                         BugReporter &BR) {
    ((const CHECKER *)checker)->checkASTDecl(cast<DECL>(D), mgr, BR);
  }

  static bool _handlesDecl(const Decl *D) {
    return isa<DECL>(D);
  }
public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForDecl(CheckerManager::CheckDeclFunc(checker,
                                                       _checkDecl<CHECKER>),
                         _handlesDecl);
  }
};

class ASTCodeBody {
  template <typename CHECKER>
  static void _checkBody(void *checker, const Decl *D, AnalysisManager& mgr,
                         BugReporter &BR) {
    ((const CHECKER *)checker)->checkASTCodeBody(D, mgr, BR);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForBody(CheckerManager::CheckDeclFunc(checker,
                                                       _checkBody<CHECKER>));
  }
};

class EndOfTranslationUnit {
  template <typename CHECKER>
  static void _checkEndOfTranslationUnit(void *checker,
                                         const TranslationUnitDecl *TU,
                                         AnalysisManager& mgr,
                                         BugReporter &BR) {
    ((const CHECKER *)checker)->checkEndOfTranslationUnit(TU, mgr, BR);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr){
    mgr._registerForEndOfTranslationUnit(
                              CheckerManager::CheckEndOfTranslationUnit(checker,
                                          _checkEndOfTranslationUnit<CHECKER>));
  }
};

template <typename STMT>
class PreStmt {
  template <typename CHECKER>
  static void _checkStmt(void *checker, const Stmt *S, CheckerContext &C) {
    ((const CHECKER *)checker)->checkPreStmt(cast<STMT>(S), C);
  }

  static bool _handlesStmt(const Stmt *S) {
    return isa<STMT>(S);
  }
public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForPreStmt(CheckerManager::CheckStmtFunc(checker,
                                                          _checkStmt<CHECKER>),
                            _handlesStmt);
  }
};

template <typename STMT>
class PostStmt {
  template <typename CHECKER>
  static void _checkStmt(void *checker, const Stmt *S, CheckerContext &C) {
    ((const CHECKER *)checker)->checkPostStmt(cast<STMT>(S), C);
  }

  static bool _handlesStmt(const Stmt *S) {
    return isa<STMT>(S);
  }
public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForPostStmt(CheckerManager::CheckStmtFunc(checker,
                                                           _checkStmt<CHECKER>),
                             _handlesStmt);
  }
};

class PreObjCMessage {
  template <typename CHECKER>
  static void _checkObjCMessage(void *checker, const ObjCMethodCall &msg,
                                CheckerContext &C) {
    ((const CHECKER *)checker)->checkPreObjCMessage(msg, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForPreObjCMessage(
     CheckerManager::CheckObjCMessageFunc(checker, _checkObjCMessage<CHECKER>));
  }
};

class ObjCMessageNil {
  template <typename CHECKER>
  static void _checkObjCMessage(void *checker, const ObjCMethodCall &msg,
                                CheckerContext &C) {
    ((const CHECKER *)checker)->checkObjCMessageNil(msg, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForObjCMessageNil(
     CheckerManager::CheckObjCMessageFunc(checker, _checkObjCMessage<CHECKER>));
  }
};

class PostObjCMessage {
  template <typename CHECKER>
  static void _checkObjCMessage(void *checker, const ObjCMethodCall &msg,
                                CheckerContext &C) {
    ((const CHECKER *)checker)->checkPostObjCMessage(msg, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForPostObjCMessage(
     CheckerManager::CheckObjCMessageFunc(checker, _checkObjCMessage<CHECKER>));
  }
};

class PreCall {
  template <typename CHECKER>
  static void _checkCall(void *checker, const CallEvent &msg,
                         CheckerContext &C) {
    ((const CHECKER *)checker)->checkPreCall(msg, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForPreCall(
     CheckerManager::CheckCallFunc(checker, _checkCall<CHECKER>));
  }
};

class PostCall {
  template <typename CHECKER>
  static void _checkCall(void *checker, const CallEvent &msg,
                         CheckerContext &C) {
    ((const CHECKER *)checker)->checkPostCall(msg, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForPostCall(
     CheckerManager::CheckCallFunc(checker, _checkCall<CHECKER>));
  }
};

class Location {
  template <typename CHECKER>
  static void _checkLocation(void *checker, SVal location, bool isLoad,
                             const Stmt *S, CheckerContext &C) {
    ((const CHECKER *)checker)->checkLocation(location, isLoad, S, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForLocation(
           CheckerManager::CheckLocationFunc(checker, _checkLocation<CHECKER>));
  }
};

class Bind {
  template <typename CHECKER>
  static void _checkBind(void *checker, SVal location, SVal val, const Stmt *S,
                         bool AtDeclInit, CheckerContext &C) {
    ((const CHECKER *)checker)->checkBind(location, val, S, AtDeclInit, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForBind(
           CheckerManager::CheckBindFunc(checker, _checkBind<CHECKER>));
  }
};

class BlockEntrance {
  template <typename CHECKER>
  static void _checkBlockEntrance(void *Checker,
                                  const clang::BlockEntrance &Entrance,
                                  CheckerContext &C) {
    ((const CHECKER *)Checker)->checkBlockEntrance(Entrance, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForBlockEntrance(CheckerManager::CheckBlockEntranceFunc(
        checker, _checkBlockEntrance<CHECKER>));
  }
};

class EndAnalysis {
  template <typename CHECKER>
  static void _checkEndAnalysis(void *checker, ExplodedGraph &G,
                                BugReporter &BR, ExprEngine &Eng) {
    ((const CHECKER *)checker)->checkEndAnalysis(G, BR, Eng);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForEndAnalysis(
     CheckerManager::CheckEndAnalysisFunc(checker, _checkEndAnalysis<CHECKER>));
  }
};

class BeginFunction {
  template <typename CHECKER>
  static void _checkBeginFunction(void *checker, CheckerContext &C) {
    ((const CHECKER *)checker)->checkBeginFunction(C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForBeginFunction(CheckerManager::CheckBeginFunctionFunc(
        checker, _checkBeginFunction<CHECKER>));
  }
};

class EndFunction {
  template <typename CHECKER>
  static void _checkEndFunction(void *checker, const ReturnStmt *RS,
                                CheckerContext &C) {
    ((const CHECKER *)checker)->checkEndFunction(RS, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForEndFunction(
     CheckerManager::CheckEndFunctionFunc(checker, _checkEndFunction<CHECKER>));
  }
};

class BranchCondition {
  template <typename CHECKER>
  static void _checkBranchCondition(void *checker, const Stmt *Condition,
                                    CheckerContext & C) {
    ((const CHECKER *)checker)->checkBranchCondition(Condition, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForBranchCondition(
      CheckerManager::CheckBranchConditionFunc(checker,
                                               _checkBranchCondition<CHECKER>));
  }
};

class NewAllocator {
  template <typename CHECKER>
  static void _checkNewAllocator(void *checker, const CXXAllocatorCall &Call,
                                 CheckerContext &C) {
    ((const CHECKER *)checker)->checkNewAllocator(Call, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForNewAllocator(
        CheckerManager::CheckNewAllocatorFunc(checker,
                                              _checkNewAllocator<CHECKER>));
  }
};

class LiveSymbols {
  template <typename CHECKER>
  static void _checkLiveSymbols(void *checker, ProgramStateRef state,
                                SymbolReaper &SR) {
    ((const CHECKER *)checker)->checkLiveSymbols(state, SR);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForLiveSymbols(
     CheckerManager::CheckLiveSymbolsFunc(checker, _checkLiveSymbols<CHECKER>));
  }
};

class DeadSymbols {
  template <typename CHECKER>
  static void _checkDeadSymbols(void *checker,
                                SymbolReaper &SR, CheckerContext &C) {
    ((const CHECKER *)checker)->checkDeadSymbols(SR, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForDeadSymbols(
     CheckerManager::CheckDeadSymbolsFunc(checker, _checkDeadSymbols<CHECKER>));
  }
};

class RegionChanges {
  template <typename CHECKER>
  static ProgramStateRef
  _checkRegionChanges(void *checker,
                      ProgramStateRef state,
                      const InvalidatedSymbols *invalidated,
                      ArrayRef<const MemRegion *> Explicits,
                      ArrayRef<const MemRegion *> Regions,
                      const LocationContext *LCtx,
                      const CallEvent *Call) {
    return ((const CHECKER *) checker)->checkRegionChanges(state, invalidated,
                                                           Explicits, Regions,
                                                           LCtx, Call);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForRegionChanges(
          CheckerManager::CheckRegionChangesFunc(checker,
                                                 _checkRegionChanges<CHECKER>));
  }
};

class PointerEscape {
  template <typename CHECKER>
  static ProgramStateRef
  _checkPointerEscape(void *Checker,
                     ProgramStateRef State,
                     const InvalidatedSymbols &Escaped,
                     const CallEvent *Call,
                     PointerEscapeKind Kind,
                     RegionAndSymbolInvalidationTraits *ETraits) {

    if (!ETraits)
      return ((const CHECKER *)Checker)->checkPointerEscape(State,
                                                            Escaped,
                                                            Call,
                                                            Kind);

    InvalidatedSymbols RegularEscape;
    for (SymbolRef Sym : Escaped)
      if (!ETraits->hasTrait(
              Sym, RegionAndSymbolInvalidationTraits::TK_PreserveContents) &&
          !ETraits->hasTrait(
              Sym, RegionAndSymbolInvalidationTraits::TK_SuppressEscape))
        RegularEscape.insert(Sym);

    if (RegularEscape.empty())
      return State;

    return ((const CHECKER *)Checker)->checkPointerEscape(State,
                                                          RegularEscape,
                                                          Call,
                                                          Kind);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForPointerEscape(
          CheckerManager::CheckPointerEscapeFunc(checker,
                                                _checkPointerEscape<CHECKER>));
  }
};

class ConstPointerEscape {
  template <typename CHECKER>
  static ProgramStateRef
  _checkConstPointerEscape(void *Checker,
                      ProgramStateRef State,
                      const InvalidatedSymbols &Escaped,
                      const CallEvent *Call,
                      PointerEscapeKind Kind,
                      RegionAndSymbolInvalidationTraits *ETraits) {

    if (!ETraits)
      return State;

    InvalidatedSymbols ConstEscape;
    for (SymbolRef Sym : Escaped) {
      if (ETraits->hasTrait(
              Sym, RegionAndSymbolInvalidationTraits::TK_PreserveContents) &&
          !ETraits->hasTrait(
              Sym, RegionAndSymbolInvalidationTraits::TK_SuppressEscape))
        ConstEscape.insert(Sym);
    }

    if (ConstEscape.empty())
      return State;

    return ((const CHECKER *)Checker)->checkConstPointerEscape(State,
                                                               ConstEscape,
                                                               Call,
                                                               Kind);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForPointerEscape(
      CheckerManager::CheckPointerEscapeFunc(checker,
                                            _checkConstPointerEscape<CHECKER>));
  }
};


template <typename EVENT>
class Event {
  template <typename CHECKER>
  static void _checkEvent(void *checker, const void *event) {
    ((const CHECKER *)checker)->checkEvent(*(const EVENT *)event);
  }
public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerListenerForEvent<EVENT>(
                 CheckerManager::CheckEventFunc(checker, _checkEvent<CHECKER>));
  }
};

} // end check namespace

namespace eval {

class Assume {
  template <typename CHECKER>
  static ProgramStateRef _evalAssume(void *checker, ProgramStateRef state,
                                     SVal cond, bool assumption) {
    return ((const CHECKER *)checker)->evalAssume(state, cond, assumption);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForEvalAssume(
                 CheckerManager::EvalAssumeFunc(checker, _evalAssume<CHECKER>));
  }
};

class Call {
  template <typename CHECKER>
  static bool _evalCall(void *checker, const CallEvent &Call,
                        CheckerContext &C) {
    return ((const CHECKER *)checker)->evalCall(Call, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForEvalCall(
                     CheckerManager::EvalCallFunc(checker, _evalCall<CHECKER>));
  }
};

} // end eval namespace

/// A `CheckerFrontend` instance is what the user recognizes as "one checker":
/// it has a public canonical name (injected from the `CheckerManager`), can be
/// enabled or disabled, can have associated checker options and can be printed
/// as the "source" of bug reports.
/// The singleton instance of a simple `Checker<...>` is-a `CheckerFrontend`
/// (for historical reasons, to preserve old straightforward code), while the
/// singleton instance of a `CheckerFamily<...>` class owns multiple
/// `CheckerFrontend` instances as data members.
/// Modeling checkers that are hidden from the user but can be enabled or
/// disabled separately (as dependencies of other checkers) are also considered
/// to be `CheckerFrontend`s.
class CheckerFrontend {
  /// The `Name` is nullopt if and only if the checker is disabled.
  std::optional<CheckerNameRef> Name;

public:
  void enable(CheckerManager &Mgr) {
    assert(!Name && "Checker part registered twice!");
    Name = Mgr.getCurrentCheckerName();
  }
  bool isEnabled() const { return Name.has_value(); }
  CheckerNameRef getName() const { return *Name; }
};

/// `CheckerBackend` is an abstract base class that serves as the common
/// ancestor of all the `Checker<...>` and `CheckerFamily<...>` classes which
/// can create `ExplodedNode`s (by acting as a `ProgramPointTag`) and can be
/// registered to handle various checker callbacks. (Moreover the debug
/// callback `printState` is also introduced here.)
class CheckerBackend : public ProgramPointTag {
public:
  /// Debug state dump callback, see CheckerManager::runCheckersForPrintState.
  /// Default implementation does nothing.
  virtual void printState(raw_ostream &Out, ProgramStateRef State,
                          const char *NL, const char *Sep) const;
};

/// The non-templated common ancestor of all the simple `Checker<...>` classes.
class CheckerBase : public CheckerFrontend, public CheckerBackend {
public:
  /// Attached to nodes created by this checker class when the ExplodedGraph is
  /// dumped for debugging.
  StringRef getDebugTag() const override;
};

/// Simple checker classes that implement one frontend (i.e. checker name)
/// should derive from this template and specify all the implemented callbacks
/// (i.e. classes like `check::PreStmt` or `eval::Call`) as template arguments
/// of `Checker`.
template <typename... CHECKs>
class Checker : public CheckerBase, public CHECKs... {
public:
  using BlockEntrance = clang::BlockEntrance;

  template <typename CHECKER>
  static void _register(CHECKER *Chk, CheckerManager &Mgr) {
    (CHECKs::_register(Chk, Mgr), ...);
  }
};

/// Checker families (where a single backend class implements multiple related
/// frontends) should derive from this template and specify all the implemented
/// callbacks (i.e. classes like `check::PreStmt` or `eval::Call`) as template
/// arguments of `FamilyChecker.`
///
/// NOTE: Classes deriving from `CheckerFamily` must implement the pure virtual
/// method `StringRef getDebugTag()` which is inherited from `ProgramPointTag`
/// and should return the name of the class as a string.
///
/// Obviously, this boilerplate is not a good thing, but unfortunately there is
/// no portable way to stringify the name of a type (e.g. class), so any
/// portable implementation of `getDebugTag` would need to take the name of
/// the class from *somewhere* where it's present as a string -- and then
/// directly placing it in a method override is much simpler than loading it
/// from `Checkers.td`.
///
/// Note that the existing `CLASS` field in `Checkers.td` is not suitable for
/// our goals, because instead of storing the same class name for each
/// frontend, in fact each frontendchecker needs to have its own unique value
/// there (to ensure that the names of the register methods are all unique).
template <typename... CHECKs>
class CheckerFamily : public CheckerBackend, public CHECKs... {
public:
  using BlockEntrance = clang::BlockEntrance;

  template <typename CHECKER>
  static void _register(CHECKER *Chk, CheckerManager &Mgr) {
    (CHECKs::_register(Chk, Mgr), ...);
  }
};

template <typename EVENT>
class EventDispatcher {
  CheckerManager *Mgr = nullptr;
public:
  EventDispatcher() = default;

  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerDispatcherForEvent<EVENT>();
    static_cast<EventDispatcher<EVENT> *>(checker)->Mgr = &mgr;
  }

  void dispatchEvent(const EVENT &event) const {
    Mgr->_dispatchEvent(event);
  }
};

/// We dereferenced a location that may be null.
struct ImplicitNullDerefEvent {
  SVal Location;
  bool IsLoad;
  ExplodedNode *SinkNode;
  BugReporter *BR;
  // When true, the dereference is in the source code directly. When false, the
  // dereference might happen later (for example pointer passed to a parameter
  // that is marked with nonnull attribute.)
  bool IsDirectDereference;

  static int Tag;
};

} // end ento namespace

} // end clang namespace

#endif
