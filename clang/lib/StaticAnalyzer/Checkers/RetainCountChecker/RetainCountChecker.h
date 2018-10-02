//==--- RetainCountChecker.h - Checks for leaks and other issues -*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the methods for RetainCountChecker, which implements
//  a reference count checker for Core Foundation and Cocoa on (Mac OS X).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_RETAINCOUNTCHECKER_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_RETAINCOUNTCHECKER_H

#include "../ClangSACheckers.h"
#include "../AllocationDiagnostics.h"
#include "RetainCountDiagnostics.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ParentMap.h"
#include "clang/Analysis/DomainSpecific/CocoaConventions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Analysis/SelectorExtras.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include "clang/StaticAnalyzer/Core/RetainSummaryManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include <cstdarg>
#include <utility>

using llvm::StrInStrNoCase;

namespace clang {
namespace ento {
namespace retaincountchecker {

/// Metadata on reference.
class RefVal {
public:
  enum Kind {
    Owned = 0, // Owning reference.
    NotOwned,  // Reference is not owned by still valid (not freed).
    Released,  // Object has been released.
    ReturnedOwned, // Returned object passes ownership to caller.
    ReturnedNotOwned, // Return object does not pass ownership to caller.
    ERROR_START,
    ErrorDeallocNotOwned, // -dealloc called on non-owned object.
    ErrorUseAfterRelease, // Object used after released.
    ErrorReleaseNotOwned, // Release of an object that was not owned.
    ERROR_LEAK_START,
    ErrorLeak,  // A memory leak due to excessive reference counts.
    ErrorLeakReturned, // A memory leak due to the returning method not having
                       // the correct naming conventions.
    ErrorOverAutorelease,
    ErrorReturnedNotOwned
  };

  /// Tracks how an object referenced by an ivar has been used.
  ///
  /// This accounts for us not knowing if an arbitrary ivar is supposed to be
  /// stored at +0 or +1.
  enum class IvarAccessHistory {
    None,
    AccessedDirectly,
    ReleasedAfterDirectAccess
  };

private:
  /// The number of outstanding retains.
  unsigned Cnt;
  /// The number of outstanding autoreleases.
  unsigned ACnt;
  /// The (static) type of the object at the time we started tracking it.
  QualType T;

  /// The current state of the object.
  ///
  /// See the RefVal::Kind enum for possible values.
  unsigned RawKind : 5;

  /// The kind of object being tracked (CF or ObjC), if known.
  ///
  /// See the RetEffect::ObjKind enum for possible values.
  unsigned RawObjectKind : 3;

  /// True if the current state and/or retain count may turn out to not be the
  /// best possible approximation of the reference counting state.
  ///
  /// If true, the checker may decide to throw away ("override") this state
  /// in favor of something else when it sees the object being used in new ways.
  ///
  /// This setting should not be propagated to state derived from this state.
  /// Once we start deriving new states, it would be inconsistent to override
  /// them.
  unsigned RawIvarAccessHistory : 2;

  RefVal(Kind k, RetEffect::ObjKind o, unsigned cnt, unsigned acnt, QualType t,
         IvarAccessHistory IvarAccess)
    : Cnt(cnt), ACnt(acnt), T(t), RawKind(static_cast<unsigned>(k)),
      RawObjectKind(static_cast<unsigned>(o)),
      RawIvarAccessHistory(static_cast<unsigned>(IvarAccess)) {
    assert(getKind() == k && "not enough bits for the kind");
    assert(getObjKind() == o && "not enough bits for the object kind");
    assert(getIvarAccessHistory() == IvarAccess && "not enough bits");
  }

public:
  Kind getKind() const { return static_cast<Kind>(RawKind); }

  RetEffect::ObjKind getObjKind() const {
    return static_cast<RetEffect::ObjKind>(RawObjectKind);
  }

  unsigned getCount() const { return Cnt; }
  unsigned getAutoreleaseCount() const { return ACnt; }
  unsigned getCombinedCounts() const { return Cnt + ACnt; }
  void clearCounts() {
    Cnt = 0;
    ACnt = 0;
  }
  void setCount(unsigned i) {
    Cnt = i;
  }
  void setAutoreleaseCount(unsigned i) {
    ACnt = i;
  }

  QualType getType() const { return T; }

  /// Returns what the analyzer knows about direct accesses to a particular
  /// instance variable.
  ///
  /// If the object with this refcount wasn't originally from an Objective-C
  /// ivar region, this should always return IvarAccessHistory::None.
  IvarAccessHistory getIvarAccessHistory() const {
    return static_cast<IvarAccessHistory>(RawIvarAccessHistory);
  }

  bool isOwned() const {
    return getKind() == Owned;
  }

  bool isNotOwned() const {
    return getKind() == NotOwned;
  }

  bool isReturnedOwned() const {
    return getKind() == ReturnedOwned;
  }

  bool isReturnedNotOwned() const {
    return getKind() == ReturnedNotOwned;
  }

  /// Create a state for an object whose lifetime is the responsibility of the
  /// current function, at least partially.
  ///
  /// Most commonly, this is an owned object with a retain count of +1.
  static RefVal makeOwned(RetEffect::ObjKind o, QualType t) {
    return RefVal(Owned, o, /*Count=*/1, 0, t, IvarAccessHistory::None);
  }

  /// Create a state for an object whose lifetime is not the responsibility of
  /// the current function.
  ///
  /// Most commonly, this is an unowned object with a retain count of +0.
  static RefVal makeNotOwned(RetEffect::ObjKind o, QualType t) {
    return RefVal(NotOwned, o, /*Count=*/0, 0, t, IvarAccessHistory::None);
  }

  RefVal operator-(size_t i) const {
    return RefVal(getKind(), getObjKind(), getCount() - i,
                  getAutoreleaseCount(), getType(), getIvarAccessHistory());
  }

  RefVal operator+(size_t i) const {
    return RefVal(getKind(), getObjKind(), getCount() + i,
                  getAutoreleaseCount(), getType(), getIvarAccessHistory());
  }

  RefVal operator^(Kind k) const {
    return RefVal(k, getObjKind(), getCount(), getAutoreleaseCount(),
                  getType(), getIvarAccessHistory());
  }

  RefVal autorelease() const {
    return RefVal(getKind(), getObjKind(), getCount(), getAutoreleaseCount()+1,
                  getType(), getIvarAccessHistory());
  }

  RefVal withIvarAccess() const {
    assert(getIvarAccessHistory() == IvarAccessHistory::None);
    return RefVal(getKind(), getObjKind(), getCount(), getAutoreleaseCount(),
                  getType(), IvarAccessHistory::AccessedDirectly);
  }

  RefVal releaseViaIvar() const {
    assert(getIvarAccessHistory() == IvarAccessHistory::AccessedDirectly);
    return RefVal(getKind(), getObjKind(), getCount(), getAutoreleaseCount(),
                  getType(), IvarAccessHistory::ReleasedAfterDirectAccess);
  }

  // Comparison, profiling, and pretty-printing.
  bool hasSameState(const RefVal &X) const {
    return getKind() == X.getKind() && Cnt == X.Cnt && ACnt == X.ACnt &&
           getIvarAccessHistory() == X.getIvarAccessHistory();
  }

  bool operator==(const RefVal& X) const {
    return T == X.T && hasSameState(X) && getObjKind() == X.getObjKind();
  }

  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.Add(T);
    ID.AddInteger(RawKind);
    ID.AddInteger(Cnt);
    ID.AddInteger(ACnt);
    ID.AddInteger(RawObjectKind);
    ID.AddInteger(RawIvarAccessHistory);
  }

  void print(raw_ostream &Out) const;
};

class RetainCountChecker
  : public Checker< check::Bind,
                    check::DeadSymbols,
                    check::EndAnalysis,
                    check::BeginFunction,
                    check::EndFunction,
                    check::PostStmt<BlockExpr>,
                    check::PostStmt<CastExpr>,
                    check::PostStmt<ObjCArrayLiteral>,
                    check::PostStmt<ObjCDictionaryLiteral>,
                    check::PostStmt<ObjCBoxedExpr>,
                    check::PostStmt<ObjCIvarRefExpr>,
                    check::PostCall,
                    check::RegionChanges,
                    eval::Assume,
                    eval::Call > {
  mutable std::unique_ptr<CFRefBug> useAfterRelease, releaseNotOwned;
  mutable std::unique_ptr<CFRefBug> deallocNotOwned;
  mutable std::unique_ptr<CFRefBug> overAutorelease, returnNotOwnedForOwned;
  mutable std::unique_ptr<CFRefBug> leakWithinFunction, leakAtReturn;

  typedef llvm::DenseMap<SymbolRef, const CheckerProgramPointTag *> SymbolTagMap;

  // This map is only used to ensure proper deletion of any allocated tags.
  mutable SymbolTagMap DeadSymbolTags;

  mutable std::unique_ptr<RetainSummaryManager> Summaries;
  mutable SummaryLogTy SummaryLog;

  AnalyzerOptions &Options;
  mutable bool ShouldResetSummaryLog;

  /// Optional setting to indicate if leak reports should include
  /// the allocation line.
  mutable bool IncludeAllocationLine;

public:
  RetainCountChecker(AnalyzerOptions &Options)
      : Options(Options), ShouldResetSummaryLog(false),
        IncludeAllocationLine(
            shouldIncludeAllocationSiteInLeakDiagnostics(Options)) {}

  ~RetainCountChecker() override { DeleteContainerSeconds(DeadSymbolTags); }

  bool shouldCheckOSObjectRetainCount() const {
    return Options.getBooleanOption("CheckOSObject", false, this);
  }

  void checkEndAnalysis(ExplodedGraph &G, BugReporter &BR,
                        ExprEngine &Eng) const {
    // FIXME: This is a hack to make sure the summary log gets cleared between
    // analyses of different code bodies.
    //
    // Why is this necessary? Because a checker's lifetime is tied to a
    // translation unit, but an ExplodedGraph's lifetime is just a code body.
    // Once in a blue moon, a new ExplodedNode will have the same address as an
    // old one with an associated summary, and the bug report visitor gets very
    // confused. (To make things worse, the summary lifetime is currently also
    // tied to a code body, so we get a crash instead of incorrect results.)
    //
    // Why is this a bad solution? Because if the lifetime of the ExplodedGraph
    // changes, things will start going wrong again. Really the lifetime of this
    // log needs to be tied to either the specific nodes in it or the entire
    // ExplodedGraph, not to a specific part of the code being analyzed.
    //
    // (Also, having stateful local data means that the same checker can't be
    // used from multiple threads, but a lot of checkers have incorrect
    // assumptions about that anyway. So that wasn't a priority at the time of
    // this fix.)
    //
    // This happens at the end of analysis, but bug reports are emitted /after/
    // this point. So we can't just clear the summary log now. Instead, we mark
    // that the next time we access the summary log, it should be cleared.

    // If we never reset the summary log during /this/ code body analysis,
    // there were no new summaries. There might still have been summaries from
    // the /last/ analysis, so clear them out to make sure the bug report
    // visitors don't get confused.
    if (ShouldResetSummaryLog)
      SummaryLog.clear();

    ShouldResetSummaryLog = !SummaryLog.empty();
  }

  CFRefBug *getLeakWithinFunctionBug(const LangOptions &LOpts) const {
    if (!leakWithinFunction)
      leakWithinFunction.reset(new Leak(this, "Leak"));
    return leakWithinFunction.get();
  }

  CFRefBug *getLeakAtReturnBug(const LangOptions &LOpts) const {
      if (!leakAtReturn)
        leakAtReturn.reset(new Leak(this, "Leak of returned object"));
      return leakAtReturn.get();
  }

  RetainSummaryManager &getSummaryManager(ASTContext &Ctx) const {
    // FIXME: We don't support ARC being turned on and off during one analysis.
    // (nor, for that matter, do we support changing ASTContexts)
    bool ARCEnabled = (bool)Ctx.getLangOpts().ObjCAutoRefCount;
    if (!Summaries) {
      Summaries.reset(new RetainSummaryManager(
          Ctx, ARCEnabled, shouldCheckOSObjectRetainCount()));
    } else {
      assert(Summaries->isARCEnabled() == ARCEnabled);
    }
    return *Summaries;
  }

  RetainSummaryManager &getSummaryManager(CheckerContext &C) const {
    return getSummaryManager(C.getASTContext());
  }

  void printState(raw_ostream &Out, ProgramStateRef State,
                  const char *NL, const char *Sep) const override;

  void checkBind(SVal loc, SVal val, const Stmt *S, CheckerContext &C) const;
  void checkPostStmt(const BlockExpr *BE, CheckerContext &C) const;
  void checkPostStmt(const CastExpr *CE, CheckerContext &C) const;

  void checkPostStmt(const ObjCArrayLiteral *AL, CheckerContext &C) const;
  void checkPostStmt(const ObjCDictionaryLiteral *DL, CheckerContext &C) const;
  void checkPostStmt(const ObjCBoxedExpr *BE, CheckerContext &C) const;

  void checkPostStmt(const ObjCIvarRefExpr *IRE, CheckerContext &C) const;

  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;

  void checkSummary(const RetainSummary &Summ, const CallEvent &Call,
                    CheckerContext &C) const;

  void processSummaryOfInlined(const RetainSummary &Summ,
                               const CallEvent &Call,
                               CheckerContext &C) const;

  bool evalCall(const CallExpr *CE, CheckerContext &C) const;

  ProgramStateRef evalAssume(ProgramStateRef state, SVal Cond,
                                 bool Assumption) const;

  ProgramStateRef
  checkRegionChanges(ProgramStateRef state,
                     const InvalidatedSymbols *invalidated,
                     ArrayRef<const MemRegion *> ExplicitRegions,
                     ArrayRef<const MemRegion *> Regions,
                     const LocationContext* LCtx,
                     const CallEvent *Call) const;

  ExplodedNode* checkReturnWithRetEffect(const ReturnStmt *S, CheckerContext &C,
                                ExplodedNode *Pred, RetEffect RE, RefVal X,
                                SymbolRef Sym, ProgramStateRef state) const;

  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
  void checkBeginFunction(CheckerContext &C) const;
  void checkEndFunction(const ReturnStmt *RS, CheckerContext &C) const;

  ProgramStateRef updateSymbol(ProgramStateRef state, SymbolRef sym,
                               RefVal V, ArgEffect E, RefVal::Kind &hasErr,
                               CheckerContext &C) const;

  void processNonLeakError(ProgramStateRef St, SourceRange ErrorRange,
                           RefVal::Kind ErrorKind, SymbolRef Sym,
                           CheckerContext &C) const;

  void processObjCLiterals(CheckerContext &C, const Expr *Ex) const;

  const ProgramPointTag *getDeadSymbolTag(SymbolRef sym) const;

  ProgramStateRef handleSymbolDeath(ProgramStateRef state,
                                    SymbolRef sid, RefVal V,
                                    SmallVectorImpl<SymbolRef> &Leaked) const;

  ProgramStateRef
  handleAutoreleaseCounts(ProgramStateRef state, ExplodedNode *Pred,
                          const ProgramPointTag *Tag, CheckerContext &Ctx,
                          SymbolRef Sym,
                          RefVal V,
                          const ReturnStmt *S=nullptr) const;

  ExplodedNode *processLeaks(ProgramStateRef state,
                             SmallVectorImpl<SymbolRef> &Leaked,
                             CheckerContext &Ctx,
                             ExplodedNode *Pred = nullptr) const;

private:
  /// Perform the necessary checks and state adjustments at the end of the
  /// function.
  /// \p S Return statement, may be null.
  ExplodedNode * processReturn(const ReturnStmt *S, CheckerContext &C) const;
};

//===----------------------------------------------------------------------===//
// RefBindings - State used to track object reference counts.
//===----------------------------------------------------------------------===//

const RefVal *getRefBinding(ProgramStateRef State, SymbolRef Sym);

ProgramStateRef setRefBinding(ProgramStateRef State, SymbolRef Sym,
                                     RefVal Val);

ProgramStateRef removeRefBinding(ProgramStateRef State, SymbolRef Sym);

/// Returns true if this stack frame is for an Objective-C method that is a
/// property getter or setter whose body has been synthesized by the analyzer.
inline bool isSynthesizedAccessor(const StackFrameContext *SFC) {
  auto Method = dyn_cast_or_null<ObjCMethodDecl>(SFC->getDecl());
  if (!Method || !Method->isPropertyAccessor())
    return false;

  return SFC->getAnalysisDeclContext()->isBodyAutosynthesized();
}

} // end namespace retaincountchecker
} // end namespace ento
} // end namespace clang

#endif
