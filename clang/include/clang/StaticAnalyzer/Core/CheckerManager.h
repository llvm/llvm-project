//===--- CheckerManager.h - Static Analyzer Checker Manager -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the Static Analyzer Checker Manager.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_CHECKERMANAGER_H
#define LLVM_CLANG_STATICANALYZER_CORE_CHECKERMANAGER_H

#include "clang/Analysis/ProgramPoint.h"
#include "clang/Basic/LangOptions.h"
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/Store.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>
#include <vector>

namespace clang {
  class Decl;
  class Stmt;
  class CallExpr;

namespace ento {
  class CheckerBase;
  class CheckerRegistry;
  class ExprEngine;
  class AnalysisManager;
  class BugReporter;
  class CheckerContext;
  class ObjCMethodCall;
  class SVal;
  class ExplodedNode;
  class ExplodedNodeSet;
  class ExplodedGraph;
  class ProgramState;
  class NodeBuilder;
  struct NodeBuilderContext;
  class MemRegion;
  class SymbolReaper;

template <typename T> class CheckerFn;

template <typename RET, typename... Ps>
class CheckerFn<RET(Ps...)> {
  typedef RET (*Func)(void *, Ps...);
  Func Fn;
public:
  CheckerBase *Checker;
  CheckerFn(CheckerBase *checker, Func fn) : Fn(fn), Checker(checker) { }
  RET operator()(Ps... ps) const {
    return Fn(Checker, ps...);
  }
};

/// \brief Describes the different reasons a pointer escapes
/// during analysis.
enum PointerEscapeKind {
  /// A pointer escapes due to binding its value to a location
  /// that the analyzer cannot track.
  PSK_EscapeOnBind,

  /// The pointer has been passed to a function call directly.
  PSK_DirectEscapeOnCall,

  /// The pointer has been passed to a function indirectly.
  /// For example, the pointer is accessible through an
  /// argument to a function.
  PSK_IndirectEscapeOnCall,

  /// The reason for pointer escape is unknown. For example, 
  /// a region containing this pointer is invalidated.
  PSK_EscapeOther
};

// This wrapper is used to ensure that only StringRefs originating from the
// CheckerRegistry are used as check names. We want to make sure all check
// name strings have a lifetime that keeps them alive at least until the path
// diagnostics have been processed.
class CheckName {
  StringRef Name;
  friend class ::clang::ento::CheckerRegistry;
  explicit CheckName(StringRef Name) : Name(Name) {}

public:
  CheckName() = default;
  StringRef getName() const { return Name; }
};

enum class ObjCMessageVisitKind {
  Pre,
  Post,
  MessageNil
};

class CheckerManager {
  const LangOptions LangOpts;
  AnalyzerOptionsRef AOptions;
  CheckName CurrentCheckName;

public:
  CheckerManager(const LangOptions &langOpts, AnalyzerOptionsRef AOptions)
      : LangOpts(langOpts), AOptions(std::move(AOptions)) {}

  ~CheckerManager();

  void setCurrentCheckName(CheckName name) { CurrentCheckName = name; }
  CheckName getCurrentCheckName() const { return CurrentCheckName; }

  bool hasPathSensitiveCheckers() const;

  void finishedCheckerRegistration();

  const LangOptions &getLangOpts() const { return LangOpts; }
  AnalyzerOptions &getAnalyzerOptions() { return *AOptions; }

  typedef CheckerBase *CheckerRef;
  typedef const void *CheckerTag;
  typedef CheckerFn<void ()> CheckerDtor;

//===----------------------------------------------------------------------===//
// registerChecker
//===----------------------------------------------------------------------===//

  /// \brief Used to register checkers.
  ///
  /// \returns a pointer to the checker object.
  template <typename CHECKER>
  CHECKER *registerChecker() {
    CheckerTag tag = getTag<CHECKER>();
    CheckerRef &ref = CheckerTags[tag];
    if (ref)
      return static_cast<CHECKER *>(ref); // already registered.

    CHECKER *checker = new CHECKER();
    checker->Name = CurrentCheckName;
    CheckerDtors.push_back(CheckerDtor(checker, destruct<CHECKER>));
    CHECKER::_register(checker, *this);
    ref = checker;
    return checker;
  }

  template <typename CHECKER>
  CHECKER *registerChecker(AnalyzerOptions &AOpts) {
    CheckerTag tag = getTag<CHECKER>();
    CheckerRef &ref = CheckerTags[tag];
    if (ref)
      return static_cast<CHECKER *>(ref); // already registered.

    CHECKER *checker = new CHECKER(AOpts);
    checker->Name = CurrentCheckName;
    CheckerDtors.push_back(CheckerDtor(checker, destruct<CHECKER>));
    CHECKER::_register(checker, *this);
    ref = checker;
    return checker;
  }

//===----------------------------------------------------------------------===//
// Functions for running checkers for AST traversing..
//===----------------------------------------------------------------------===//

  /// \brief Run checkers handling Decls.
  void runCheckersOnASTDecl(const Decl *D, AnalysisManager& mgr,
                            BugReporter &BR);

  /// \brief Run checkers handling Decls containing a Stmt body.
  void runCheckersOnASTBody(const Decl *D, AnalysisManager& mgr,
                            BugReporter &BR);

//===----------------------------------------------------------------------===//
// Functions for running checkers for path-sensitive checking.
//===----------------------------------------------------------------------===//

  /// \brief Run checkers for pre-visiting Stmts.
  ///
  /// The notification is performed for every explored CFGElement, which does
  /// not include the control flow statements such as IfStmt.
  ///
  /// \sa runCheckersForBranchCondition, runCheckersForPostStmt
  void runCheckersForPreStmt(ExplodedNodeSet &Dst,
                             const ExplodedNodeSet &Src,
                             const Stmt *S,
                             ExprEngine &Eng) {
    runCheckersForStmt(/*isPreVisit=*/true, Dst, Src, S, Eng);
  }

  /// \brief Run checkers for post-visiting Stmts.
  ///
  /// The notification is performed for every explored CFGElement, which does
  /// not include the control flow statements such as IfStmt.
  ///
  /// \sa runCheckersForBranchCondition, runCheckersForPreStmt
  void runCheckersForPostStmt(ExplodedNodeSet &Dst,
                              const ExplodedNodeSet &Src,
                              const Stmt *S,
                              ExprEngine &Eng,
                              bool wasInlined = false) {
    runCheckersForStmt(/*isPreVisit=*/false, Dst, Src, S, Eng, wasInlined);
  }

  /// \brief Run checkers for visiting Stmts.
  void runCheckersForStmt(bool isPreVisit,
                          ExplodedNodeSet &Dst, const ExplodedNodeSet &Src,
                          const Stmt *S, ExprEngine &Eng,
                          bool wasInlined = false);

  /// \brief Run checkers for pre-visiting obj-c messages.
  void runCheckersForPreObjCMessage(ExplodedNodeSet &Dst,
                                    const ExplodedNodeSet &Src,
                                    const ObjCMethodCall &msg,
                                    ExprEngine &Eng) {
    runCheckersForObjCMessage(ObjCMessageVisitKind::Pre, Dst, Src, msg, Eng);
  }

  /// \brief Run checkers for post-visiting obj-c messages.
  void runCheckersForPostObjCMessage(ExplodedNodeSet &Dst,
                                     const ExplodedNodeSet &Src,
                                     const ObjCMethodCall &msg,
                                     ExprEngine &Eng,
                                     bool wasInlined = false) {
    runCheckersForObjCMessage(ObjCMessageVisitKind::Post, Dst, Src, msg, Eng,
                              wasInlined);
  }

  /// \brief Run checkers for visiting an obj-c message to nil.
  void runCheckersForObjCMessageNil(ExplodedNodeSet &Dst,
                                    const ExplodedNodeSet &Src,
                                    const ObjCMethodCall &msg,
                                    ExprEngine &Eng) {
    runCheckersForObjCMessage(ObjCMessageVisitKind::MessageNil, Dst, Src, msg,
                              Eng);
  }


  /// \brief Run checkers for visiting obj-c messages.
  void runCheckersForObjCMessage(ObjCMessageVisitKind visitKind,
                                 ExplodedNodeSet &Dst,
                                 const ExplodedNodeSet &Src,
                                 const ObjCMethodCall &msg, ExprEngine &Eng,
                                 bool wasInlined = false);

  /// \brief Run checkers for pre-visiting obj-c messages.
  void runCheckersForPreCall(ExplodedNodeSet &Dst, const ExplodedNodeSet &Src,
                             const CallEvent &Call, ExprEngine &Eng) {
    runCheckersForCallEvent(/*isPreVisit=*/true, Dst, Src, Call, Eng);
  }

  /// \brief Run checkers for post-visiting obj-c messages.
  void runCheckersForPostCall(ExplodedNodeSet &Dst, const ExplodedNodeSet &Src,
                              const CallEvent &Call, ExprEngine &Eng,
                              bool wasInlined = false) {
    runCheckersForCallEvent(/*isPreVisit=*/false, Dst, Src, Call, Eng,
                            wasInlined);
  }

  /// \brief Run checkers for visiting obj-c messages.
  void runCheckersForCallEvent(bool isPreVisit, ExplodedNodeSet &Dst,
                               const ExplodedNodeSet &Src,
                               const CallEvent &Call, ExprEngine &Eng,
                               bool wasInlined = false);

  /// \brief Run checkers for load/store of a location.
  void runCheckersForLocation(ExplodedNodeSet &Dst,
                              const ExplodedNodeSet &Src,
                              SVal location,
                              bool isLoad,
                              const Stmt *NodeEx,
                              const Stmt *BoundEx,
                              ExprEngine &Eng);

  /// \brief Run checkers for binding of a value to a location.
  void runCheckersForBind(ExplodedNodeSet &Dst,
                          const ExplodedNodeSet &Src,
                          SVal location, SVal val,
                          const Stmt *S, ExprEngine &Eng,
                          const ProgramPoint &PP);

  /// \brief Run checkers for end of analysis.
  void runCheckersForEndAnalysis(ExplodedGraph &G, BugReporter &BR,
                                 ExprEngine &Eng);

  /// \brief Run checkers on begining of function.
  void runCheckersForBeginFunction(ExplodedNodeSet &Dst,
                                   const BlockEdge &L,
                                   ExplodedNode *Pred,
                                   ExprEngine &Eng);

  /// \brief Run checkers on end of function.
  void runCheckersForEndFunction(NodeBuilderContext &BC,
                                 ExplodedNodeSet &Dst,
                                 ExplodedNode *Pred,
                                 ExprEngine &Eng);

  /// \brief Run checkers for branch condition.
  void runCheckersForBranchCondition(const Stmt *condition,
                                     ExplodedNodeSet &Dst, ExplodedNode *Pred,
                                     ExprEngine &Eng);

  /// \brief Run checkers for live symbols.
  ///
  /// Allows modifying SymbolReaper object. For example, checkers can explicitly
  /// register symbols of interest as live. These symbols will not be marked
  /// dead and removed.
  void runCheckersForLiveSymbols(ProgramStateRef state,
                                 SymbolReaper &SymReaper);

  /// \brief Run checkers for dead symbols.
  ///
  /// Notifies checkers when symbols become dead. For example, this allows
  /// checkers to aggressively clean up/reduce the checker state and produce
  /// precise diagnostics.
  void runCheckersForDeadSymbols(ExplodedNodeSet &Dst,
                                 const ExplodedNodeSet &Src,
                                 SymbolReaper &SymReaper, const Stmt *S,
                                 ExprEngine &Eng,
                                 ProgramPoint::Kind K);

  /// \brief True if at least one checker wants to check region changes.
  bool wantsRegionChangeUpdate(ProgramStateRef state);

  /// \brief Run checkers for region changes.
  ///
  /// This corresponds to the check::RegionChanges callback.
  /// \param state The current program state.
  /// \param invalidated A set of all symbols potentially touched by the change.
  /// \param ExplicitRegions The regions explicitly requested for invalidation.
  ///   For example, in the case of a function call, these would be arguments.
  /// \param Regions The transitive closure of accessible regions,
  ///   i.e. all regions that may have been touched by this change.
  /// \param Call The call expression wrapper if the regions are invalidated
  ///   by a call.
  ProgramStateRef
  runCheckersForRegionChanges(ProgramStateRef state,
                              const InvalidatedSymbols *invalidated,
                              ArrayRef<const MemRegion *> ExplicitRegions,
                              ArrayRef<const MemRegion *> Regions,
                              const CallEvent *Call);

  /// \brief Run checkers when pointers escape.
  ///
  /// This notifies the checkers about pointer escape, which occurs whenever
  /// the analyzer cannot track the symbol any more. For example, as a
  /// result of assigning a pointer into a global or when it's passed to a 
  /// function call the analyzer cannot model.
  /// 
  /// \param State The state at the point of escape.
  /// \param Escaped The list of escaped symbols.
  /// \param Call The corresponding CallEvent, if the symbols escape as 
  ///        parameters to the given call.
  /// \param Kind The reason of pointer escape.
  /// \param ITraits Information about invalidation for a particular 
  ///        region/symbol.
  /// \returns Checkers can modify the state by returning a new one.
  ProgramStateRef 
  runCheckersForPointerEscape(ProgramStateRef State,
                              const InvalidatedSymbols &Escaped,
                              const CallEvent *Call,
                              PointerEscapeKind Kind,
                             RegionAndSymbolInvalidationTraits *ITraits);

  /// \brief Run checkers for handling assumptions on symbolic values.
  ProgramStateRef runCheckersForEvalAssume(ProgramStateRef state,
                                           SVal Cond, bool Assumption);

  /// \brief Run checkers for evaluating a call.
  ///
  /// Warning: Currently, the CallEvent MUST come from a CallExpr!
  void runCheckersForEvalCall(ExplodedNodeSet &Dst,
                              const ExplodedNodeSet &Src,
                              const CallEvent &CE, ExprEngine &Eng);
  
  /// \brief Run checkers for the entire Translation Unit.
  void runCheckersOnEndOfTranslationUnit(const TranslationUnitDecl *TU,
                                         AnalysisManager &mgr,
                                         BugReporter &BR);

  /// \brief Run checkers for debug-printing a ProgramState.
  ///
  /// Unlike most other callbacks, any checker can simply implement the virtual
  /// method CheckerBase::printState if it has custom data to print.
  /// \param Out The output stream
  /// \param State The state being printed
  /// \param NL The preferred representation of a newline.
  /// \param Sep The preferred separator between different kinds of data.
  void runCheckersForPrintState(raw_ostream &Out, ProgramStateRef State,
                                const char *NL, const char *Sep);

//===----------------------------------------------------------------------===//
// Internal registration functions for AST traversing.
//===----------------------------------------------------------------------===//

  // Functions used by the registration mechanism, checkers should not touch
  // these directly.

  typedef CheckerFn<void (const Decl *, AnalysisManager&, BugReporter &)>
      CheckDeclFunc;

  typedef bool (*HandlesDeclFunc)(const Decl *D);
  void _registerForDecl(CheckDeclFunc checkfn, HandlesDeclFunc isForDeclFn);

  void _registerForBody(CheckDeclFunc checkfn);

//===----------------------------------------------------------------------===//
// Internal registration functions for path-sensitive checking.
//===----------------------------------------------------------------------===//

  typedef CheckerFn<void (const Stmt *, CheckerContext &)> CheckStmtFunc;
  
  typedef CheckerFn<void (const ObjCMethodCall &, CheckerContext &)>
      CheckObjCMessageFunc;

  typedef CheckerFn<void (const CallEvent &, CheckerContext &)>
      CheckCallFunc;
  
  typedef CheckerFn<void (const SVal &location, bool isLoad,
                          const Stmt *S,
                          CheckerContext &)>
      CheckLocationFunc;
  
  typedef CheckerFn<void (const SVal &location, const SVal &val, 
                          const Stmt *S, CheckerContext &)> 
      CheckBindFunc;
  
  typedef CheckerFn<void (ExplodedGraph &, BugReporter &, ExprEngine &)>
      CheckEndAnalysisFunc;

  typedef CheckerFn<void (CheckerContext &)>
      CheckBeginFunctionFunc;

  typedef CheckerFn<void (CheckerContext &)>
      CheckEndFunctionFunc;
  
  typedef CheckerFn<void (const Stmt *, CheckerContext &)>
      CheckBranchConditionFunc;
  
  typedef CheckerFn<void (SymbolReaper &, CheckerContext &)>
      CheckDeadSymbolsFunc;
  
  typedef CheckerFn<void (ProgramStateRef,SymbolReaper &)> CheckLiveSymbolsFunc;
  
  typedef CheckerFn<ProgramStateRef (ProgramStateRef,
                                const InvalidatedSymbols *symbols,
                                ArrayRef<const MemRegion *> ExplicitRegions,
                                ArrayRef<const MemRegion *> Regions,
                                const CallEvent *Call)>
      CheckRegionChangesFunc;
  
  typedef CheckerFn<bool (ProgramStateRef)> WantsRegionChangeUpdateFunc;

  typedef CheckerFn<ProgramStateRef (ProgramStateRef,
                                     const InvalidatedSymbols &Escaped,
                                     const CallEvent *Call,
                                     PointerEscapeKind Kind,
                                     RegionAndSymbolInvalidationTraits *ITraits)>
      CheckPointerEscapeFunc;
  
  typedef CheckerFn<ProgramStateRef (ProgramStateRef,
                                          const SVal &cond, bool assumption)>
      EvalAssumeFunc;
  
  typedef CheckerFn<bool (const CallExpr *, CheckerContext &)>
      EvalCallFunc;

  typedef CheckerFn<void (const TranslationUnitDecl *,
                          AnalysisManager&, BugReporter &)>
      CheckEndOfTranslationUnit;

  typedef bool (*HandlesStmtFunc)(const Stmt *D);
  void _registerForPreStmt(CheckStmtFunc checkfn,
                           HandlesStmtFunc isForStmtFn);
  void _registerForPostStmt(CheckStmtFunc checkfn,
                            HandlesStmtFunc isForStmtFn);

  void _registerForPreObjCMessage(CheckObjCMessageFunc checkfn);
  void _registerForPostObjCMessage(CheckObjCMessageFunc checkfn);

  void _registerForObjCMessageNil(CheckObjCMessageFunc checkfn);

  void _registerForPreCall(CheckCallFunc checkfn);
  void _registerForPostCall(CheckCallFunc checkfn);

  void _registerForLocation(CheckLocationFunc checkfn);

  void _registerForBind(CheckBindFunc checkfn);

  void _registerForEndAnalysis(CheckEndAnalysisFunc checkfn);

  void _registerForBeginFunction(CheckEndFunctionFunc checkfn);
  void _registerForEndFunction(CheckEndFunctionFunc checkfn);

  void _registerForBranchCondition(CheckBranchConditionFunc checkfn);

  void _registerForLiveSymbols(CheckLiveSymbolsFunc checkfn);

  void _registerForDeadSymbols(CheckDeadSymbolsFunc checkfn);

  void _registerForRegionChanges(CheckRegionChangesFunc checkfn,
                                 WantsRegionChangeUpdateFunc wantUpdateFn);

  void _registerForPointerEscape(CheckPointerEscapeFunc checkfn);

  void _registerForConstPointerEscape(CheckPointerEscapeFunc checkfn);

  void _registerForEvalAssume(EvalAssumeFunc checkfn);

  void _registerForEvalCall(EvalCallFunc checkfn);

  void _registerForEndOfTranslationUnit(CheckEndOfTranslationUnit checkfn);

//===----------------------------------------------------------------------===//
// Internal registration functions for events.
//===----------------------------------------------------------------------===//

  typedef void *EventTag;
  typedef CheckerFn<void (const void *event)> CheckEventFunc;

  template <typename EVENT>
  void _registerListenerForEvent(CheckEventFunc checkfn) {
    EventInfo &info = Events[getTag<EVENT>()];
    info.Checkers.push_back(checkfn);    
  }

  template <typename EVENT>
  void _registerDispatcherForEvent() {
    EventInfo &info = Events[getTag<EVENT>()];
    info.HasDispatcher = true;
  }

  template <typename EVENT>
  void _dispatchEvent(const EVENT &event) const {
    EventsTy::const_iterator I = Events.find(getTag<EVENT>());
    if (I == Events.end())
      return;
    const EventInfo &info = I->second;
    for (unsigned i = 0, e = info.Checkers.size(); i != e; ++i)
      info.Checkers[i](&event);
  }

//===----------------------------------------------------------------------===//
// Implementation details.
//===----------------------------------------------------------------------===//

private:
  template <typename CHECKER>
  static void destruct(void *obj) { delete static_cast<CHECKER *>(obj); }

  template <typename T>
  static void *getTag() { static int tag; return &tag; }

  llvm::DenseMap<CheckerTag, CheckerRef> CheckerTags;

  std::vector<CheckerDtor> CheckerDtors;

  struct DeclCheckerInfo {
    CheckDeclFunc CheckFn;
    HandlesDeclFunc IsForDeclFn;
  };
  std::vector<DeclCheckerInfo> DeclCheckers;

  std::vector<CheckDeclFunc> BodyCheckers;

  typedef SmallVector<CheckDeclFunc, 4> CachedDeclCheckers;
  typedef llvm::DenseMap<unsigned, CachedDeclCheckers> CachedDeclCheckersMapTy;
  CachedDeclCheckersMapTy CachedDeclCheckersMap;

  struct StmtCheckerInfo {
    CheckStmtFunc CheckFn;
    HandlesStmtFunc IsForStmtFn;
    bool IsPreVisit;
  };
  std::vector<StmtCheckerInfo> StmtCheckers;

  typedef SmallVector<CheckStmtFunc, 4> CachedStmtCheckers;
  typedef llvm::DenseMap<unsigned, CachedStmtCheckers> CachedStmtCheckersMapTy;
  CachedStmtCheckersMapTy CachedStmtCheckersMap;

  const CachedStmtCheckers &getCachedStmtCheckersFor(const Stmt *S,
                                                     bool isPreVisit);

  /// Returns the checkers that have registered for callbacks of the
  /// given \p Kind.
  const std::vector<CheckObjCMessageFunc> &
  getObjCMessageCheckers(ObjCMessageVisitKind Kind);

  std::vector<CheckObjCMessageFunc> PreObjCMessageCheckers;
  std::vector<CheckObjCMessageFunc> PostObjCMessageCheckers;
  std::vector<CheckObjCMessageFunc> ObjCMessageNilCheckers;

  std::vector<CheckCallFunc> PreCallCheckers;
  std::vector<CheckCallFunc> PostCallCheckers;

  std::vector<CheckLocationFunc> LocationCheckers;

  std::vector<CheckBindFunc> BindCheckers;

  std::vector<CheckEndAnalysisFunc> EndAnalysisCheckers;

  std::vector<CheckBeginFunctionFunc> BeginFunctionCheckers;
  std::vector<CheckEndFunctionFunc> EndFunctionCheckers;

  std::vector<CheckBranchConditionFunc> BranchConditionCheckers;

  std::vector<CheckLiveSymbolsFunc> LiveSymbolsCheckers;

  std::vector<CheckDeadSymbolsFunc> DeadSymbolsCheckers;

  struct RegionChangesCheckerInfo {
    CheckRegionChangesFunc CheckFn;
    WantsRegionChangeUpdateFunc WantUpdateFn;
  };
  std::vector<RegionChangesCheckerInfo> RegionChangesCheckers;

  std::vector<CheckPointerEscapeFunc> PointerEscapeCheckers;

  std::vector<EvalAssumeFunc> EvalAssumeCheckers;

  std::vector<EvalCallFunc> EvalCallCheckers;

  std::vector<CheckEndOfTranslationUnit> EndOfTranslationUnitCheckers;

  struct EventInfo {
    SmallVector<CheckEventFunc, 4> Checkers;
    bool HasDispatcher;
    EventInfo() : HasDispatcher(false) { }
  };
  
  typedef llvm::DenseMap<EventTag, EventInfo> EventsTy;
  EventsTy Events;
};

} // end ento namespace

} // end clang namespace

#endif
