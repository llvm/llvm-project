//==-- RetainCountChecker.cpp - Checks for leaks and other issues -*- C++ -*--//
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

#include "ClangSACheckers.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Analysis/DomainSpecific/CocoaConventions.h"
#include "clang/AST/ParentMap.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include <cstdarg>

using namespace clang;
using namespace ento;
using llvm::StrInStrNoCase;

//===----------------------------------------------------------------------===//
// Primitives used for constructing summaries for function/method calls.
//===----------------------------------------------------------------------===//

/// ArgEffect is used to summarize a function/method call's effect on a
/// particular argument.
enum ArgEffect { DoNothing, Autorelease, Dealloc, DecRef, DecRefMsg,
                 DecRefBridgedTransfered,
                 IncRefMsg, IncRef, MakeCollectable, MayEscape,
                 NewAutoreleasePool,

                 // Stop tracking the argument - the effect of the call is
                 // unknown.
                 StopTracking,

                 // In some cases, we obtain a better summary for this checker
                 // by looking at the call site than by inlining the function.
                 // Signifies that we should stop tracking the symbol even if
                 // the function is inlined.
                 StopTrackingHard,

                 // The function decrements the reference count and the checker 
                 // should stop tracking the argument.
                 DecRefAndStopTrackingHard, DecRefMsgAndStopTrackingHard
               };

namespace llvm {
template <> struct FoldingSetTrait<ArgEffect> {
static inline void Profile(const ArgEffect X, FoldingSetNodeID& ID) {
  ID.AddInteger((unsigned) X);
}
};
} // end llvm namespace

/// ArgEffects summarizes the effects of a function/method call on all of
/// its arguments.
typedef llvm::ImmutableMap<unsigned,ArgEffect> ArgEffects;

namespace {

///  RetEffect is used to summarize a function/method call's behavior with
///  respect to its return value.
class RetEffect {
public:
  enum Kind { NoRet, OwnedSymbol, OwnedAllocatedSymbol,
              NotOwnedSymbol, GCNotOwnedSymbol, ARCNotOwnedSymbol,
              OwnedWhenTrackedReceiver,
              // Treat this function as returning a non-tracked symbol even if
              // the function has been inlined. This is used where the call
              // site summary is more presise than the summary indirectly produced 
              // by inlining the function
              NoRetHard
            };

  enum ObjKind { CF, ObjC, AnyObj };

private:
  Kind K;
  ObjKind O;

  RetEffect(Kind k, ObjKind o = AnyObj) : K(k), O(o) {}

public:
  Kind getKind() const { return K; }

  ObjKind getObjKind() const { return O; }

  bool isOwned() const {
    return K == OwnedSymbol || K == OwnedAllocatedSymbol ||
           K == OwnedWhenTrackedReceiver;
  }

  bool operator==(const RetEffect &Other) const {
    return K == Other.K && O == Other.O;
  }

  static RetEffect MakeOwnedWhenTrackedReceiver() {
    return RetEffect(OwnedWhenTrackedReceiver, ObjC);
  }

  static RetEffect MakeOwned(ObjKind o, bool isAllocated = false) {
    return RetEffect(isAllocated ? OwnedAllocatedSymbol : OwnedSymbol, o);
  }
  static RetEffect MakeNotOwned(ObjKind o) {
    return RetEffect(NotOwnedSymbol, o);
  }
  static RetEffect MakeGCNotOwned() {
    return RetEffect(GCNotOwnedSymbol, ObjC);
  }
  static RetEffect MakeARCNotOwned() {
    return RetEffect(ARCNotOwnedSymbol, ObjC);
  }
  static RetEffect MakeNoRet() {
    return RetEffect(NoRet);
  }
  static RetEffect MakeNoRetHard() {
    return RetEffect(NoRetHard);
  }

  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger((unsigned) K);
    ID.AddInteger((unsigned) O);
  }
};

//===----------------------------------------------------------------------===//
// Reference-counting logic (typestate + counts).
//===----------------------------------------------------------------------===//

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
    ErrorDeallocGC, // Calling -dealloc with GC enabled.
    ErrorUseAfterRelease, // Object used after released.
    ErrorReleaseNotOwned, // Release of an object that was not owned.
    ERROR_LEAK_START,
    ErrorLeak,  // A memory leak due to excessive reference counts.
    ErrorLeakReturned, // A memory leak due to the returning method not having
                       // the correct naming conventions.
    ErrorGCLeakReturned,
    ErrorOverAutorelease,
    ErrorReturnedNotOwned
  };

private:
  Kind kind;
  RetEffect::ObjKind okind;
  unsigned Cnt;
  unsigned ACnt;
  QualType T;

  RefVal(Kind k, RetEffect::ObjKind o, unsigned cnt, unsigned acnt, QualType t)
  : kind(k), okind(o), Cnt(cnt), ACnt(acnt), T(t) {}

public:
  Kind getKind() const { return kind; }

  RetEffect::ObjKind getObjKind() const { return okind; }

  unsigned getCount() const { return Cnt; }
  unsigned getAutoreleaseCount() const { return ACnt; }
  unsigned getCombinedCounts() const { return Cnt + ACnt; }
  void clearCounts() { Cnt = 0; ACnt = 0; }
  void setCount(unsigned i) { Cnt = i; }
  void setAutoreleaseCount(unsigned i) { ACnt = i; }

  QualType getType() const { return T; }

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

  static RefVal makeOwned(RetEffect::ObjKind o, QualType t,
                          unsigned Count = 1) {
    return RefVal(Owned, o, Count, 0, t);
  }

  static RefVal makeNotOwned(RetEffect::ObjKind o, QualType t,
                             unsigned Count = 0) {
    return RefVal(NotOwned, o, Count, 0, t);
  }

  // Comparison, profiling, and pretty-printing.

  bool operator==(const RefVal& X) const {
    return kind == X.kind && Cnt == X.Cnt && T == X.T && ACnt == X.ACnt;
  }

  RefVal operator-(size_t i) const {
    return RefVal(getKind(), getObjKind(), getCount() - i,
                  getAutoreleaseCount(), getType());
  }

  RefVal operator+(size_t i) const {
    return RefVal(getKind(), getObjKind(), getCount() + i,
                  getAutoreleaseCount(), getType());
  }

  RefVal operator^(Kind k) const {
    return RefVal(k, getObjKind(), getCount(), getAutoreleaseCount(),
                  getType());
  }

  RefVal autorelease() const {
    return RefVal(getKind(), getObjKind(), getCount(), getAutoreleaseCount()+1,
                  getType());
  }

  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger((unsigned) kind);
    ID.AddInteger(Cnt);
    ID.AddInteger(ACnt);
    ID.Add(T);
  }

  void print(raw_ostream &Out) const;
};

void RefVal::print(raw_ostream &Out) const {
  if (!T.isNull())
    Out << "Tracked " << T.getAsString() << '/';

  switch (getKind()) {
    default: llvm_unreachable("Invalid RefVal kind");
    case Owned: {
      Out << "Owned";
      unsigned cnt = getCount();
      if (cnt) Out << " (+ " << cnt << ")";
      break;
    }

    case NotOwned: {
      Out << "NotOwned";
      unsigned cnt = getCount();
      if (cnt) Out << " (+ " << cnt << ")";
      break;
    }

    case ReturnedOwned: {
      Out << "ReturnedOwned";
      unsigned cnt = getCount();
      if (cnt) Out << " (+ " << cnt << ")";
      break;
    }

    case ReturnedNotOwned: {
      Out << "ReturnedNotOwned";
      unsigned cnt = getCount();
      if (cnt) Out << " (+ " << cnt << ")";
      break;
    }

    case Released:
      Out << "Released";
      break;

    case ErrorDeallocGC:
      Out << "-dealloc (GC)";
      break;

    case ErrorDeallocNotOwned:
      Out << "-dealloc (not-owned)";
      break;

    case ErrorLeak:
      Out << "Leaked";
      break;

    case ErrorLeakReturned:
      Out << "Leaked (Bad naming)";
      break;

    case ErrorGCLeakReturned:
      Out << "Leaked (GC-ed at return)";
      break;

    case ErrorUseAfterRelease:
      Out << "Use-After-Release [ERROR]";
      break;

    case ErrorReleaseNotOwned:
      Out << "Release of Not-Owned [ERROR]";
      break;

    case RefVal::ErrorOverAutorelease:
      Out << "Over autoreleased";
      break;

    case RefVal::ErrorReturnedNotOwned:
      Out << "Non-owned object returned instead of owned";
      break;
  }

  if (ACnt) {
    Out << " [ARC +" << ACnt << ']';
  }
}
} //end anonymous namespace

//===----------------------------------------------------------------------===//
// RefBindings - State used to track object reference counts.
//===----------------------------------------------------------------------===//

REGISTER_MAP_WITH_PROGRAMSTATE(RefBindings, SymbolRef, RefVal)

static inline const RefVal *getRefBinding(ProgramStateRef State,
                                          SymbolRef Sym) {
  return State->get<RefBindings>(Sym);
}

static inline ProgramStateRef setRefBinding(ProgramStateRef State,
                                            SymbolRef Sym, RefVal Val) {
  return State->set<RefBindings>(Sym, Val);
}

static ProgramStateRef removeRefBinding(ProgramStateRef State, SymbolRef Sym) {
  return State->remove<RefBindings>(Sym);
}

//===----------------------------------------------------------------------===//
// Function/Method behavior summaries.
//===----------------------------------------------------------------------===//

namespace {
class RetainSummary {
  /// Args - a map of (index, ArgEffect) pairs, where index
  ///  specifies the argument (starting from 0).  This can be sparsely
  ///  populated; arguments with no entry in Args use 'DefaultArgEffect'.
  ArgEffects Args;

  /// DefaultArgEffect - The default ArgEffect to apply to arguments that
  ///  do not have an entry in Args.
  ArgEffect DefaultArgEffect;

  /// Receiver - If this summary applies to an Objective-C message expression,
  ///  this is the effect applied to the state of the receiver.
  ArgEffect Receiver;

  /// Ret - The effect on the return value.  Used to indicate if the
  ///  function/method call returns a new tracked symbol.
  RetEffect Ret;

public:
  RetainSummary(ArgEffects A, RetEffect R, ArgEffect defaultEff,
                ArgEffect ReceiverEff)
    : Args(A), DefaultArgEffect(defaultEff), Receiver(ReceiverEff), Ret(R) {}

  /// getArg - Return the argument effect on the argument specified by
  ///  idx (starting from 0).
  ArgEffect getArg(unsigned idx) const {
    if (const ArgEffect *AE = Args.lookup(idx))
      return *AE;

    return DefaultArgEffect;
  }
  
  void addArg(ArgEffects::Factory &af, unsigned idx, ArgEffect e) {
    Args = af.add(Args, idx, e);
  }

  /// setDefaultArgEffect - Set the default argument effect.
  void setDefaultArgEffect(ArgEffect E) {
    DefaultArgEffect = E;
  }

  /// getRetEffect - Returns the effect on the return value of the call.
  RetEffect getRetEffect() const { return Ret; }

  /// setRetEffect - Set the effect of the return value of the call.
  void setRetEffect(RetEffect E) { Ret = E; }

  
  /// Sets the effect on the receiver of the message.
  void setReceiverEffect(ArgEffect e) { Receiver = e; }
  
  /// getReceiverEffect - Returns the effect on the receiver of the call.
  ///  This is only meaningful if the summary applies to an ObjCMessageExpr*.
  ArgEffect getReceiverEffect() const { return Receiver; }

  /// Test if two retain summaries are identical. Note that merely equivalent
  /// summaries are not necessarily identical (for example, if an explicit 
  /// argument effect matches the default effect).
  bool operator==(const RetainSummary &Other) const {
    return Args == Other.Args && DefaultArgEffect == Other.DefaultArgEffect &&
           Receiver == Other.Receiver && Ret == Other.Ret;
  }

  /// Profile this summary for inclusion in a FoldingSet.
  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.Add(Args);
    ID.Add(DefaultArgEffect);
    ID.Add(Receiver);
    ID.Add(Ret);
  }

  /// A retain summary is simple if it has no ArgEffects other than the default.
  bool isSimple() const {
    return Args.isEmpty();
  }

private:
  ArgEffects getArgEffects() const { return Args; }
  ArgEffect getDefaultArgEffect() const { return DefaultArgEffect; }

  friend class RetainSummaryManager;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Data structures for constructing summaries.
//===----------------------------------------------------------------------===//

namespace {
class ObjCSummaryKey {
  IdentifierInfo* II;
  Selector S;
public:
  ObjCSummaryKey(IdentifierInfo* ii, Selector s)
    : II(ii), S(s) {}

  ObjCSummaryKey(const ObjCInterfaceDecl *d, Selector s)
    : II(d ? d->getIdentifier() : 0), S(s) {}

  ObjCSummaryKey(Selector s)
    : II(0), S(s) {}

  IdentifierInfo *getIdentifier() const { return II; }
  Selector getSelector() const { return S; }
};
}

namespace llvm {
template <> struct DenseMapInfo<ObjCSummaryKey> {
  static inline ObjCSummaryKey getEmptyKey() {
    return ObjCSummaryKey(DenseMapInfo<IdentifierInfo*>::getEmptyKey(),
                          DenseMapInfo<Selector>::getEmptyKey());
  }

  static inline ObjCSummaryKey getTombstoneKey() {
    return ObjCSummaryKey(DenseMapInfo<IdentifierInfo*>::getTombstoneKey(),
                          DenseMapInfo<Selector>::getTombstoneKey());
  }

  static unsigned getHashValue(const ObjCSummaryKey &V) {
    typedef std::pair<IdentifierInfo*, Selector> PairTy;
    return DenseMapInfo<PairTy>::getHashValue(PairTy(V.getIdentifier(),
                                                     V.getSelector()));
  }

  static bool isEqual(const ObjCSummaryKey& LHS, const ObjCSummaryKey& RHS) {
    return LHS.getIdentifier() == RHS.getIdentifier() &&
           LHS.getSelector() == RHS.getSelector();
  }

};
template <>
struct isPodLike<ObjCSummaryKey> { static const bool value = true; };
} // end llvm namespace

namespace {
class ObjCSummaryCache {
  typedef llvm::DenseMap<ObjCSummaryKey, const RetainSummary *> MapTy;
  MapTy M;
public:
  ObjCSummaryCache() {}

  const RetainSummary * find(const ObjCInterfaceDecl *D, Selector S) {
    // Do a lookup with the (D,S) pair.  If we find a match return
    // the iterator.
    ObjCSummaryKey K(D, S);
    MapTy::iterator I = M.find(K);

    if (I != M.end())
      return I->second;
    if (!D)
      return NULL;

    // Walk the super chain.  If we find a hit with a parent, we'll end
    // up returning that summary.  We actually allow that key (null,S), as
    // we cache summaries for the null ObjCInterfaceDecl* to allow us to
    // generate initial summaries without having to worry about NSObject
    // being declared.
    // FIXME: We may change this at some point.
    for (ObjCInterfaceDecl *C=D->getSuperClass() ;; C=C->getSuperClass()) {
      if ((I = M.find(ObjCSummaryKey(C, S))) != M.end())
        break;

      if (!C)
        return NULL;
    }

    // Cache the summary with original key to make the next lookup faster
    // and return the iterator.
    const RetainSummary *Summ = I->second;
    M[K] = Summ;
    return Summ;
  }

  const RetainSummary *find(IdentifierInfo* II, Selector S) {
    // FIXME: Class method lookup.  Right now we dont' have a good way
    // of going between IdentifierInfo* and the class hierarchy.
    MapTy::iterator I = M.find(ObjCSummaryKey(II, S));

    if (I == M.end())
      I = M.find(ObjCSummaryKey(S));

    return I == M.end() ? NULL : I->second;
  }

  const RetainSummary *& operator[](ObjCSummaryKey K) {
    return M[K];
  }

  const RetainSummary *& operator[](Selector S) {
    return M[ ObjCSummaryKey(S) ];
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Data structures for managing collections of summaries.
//===----------------------------------------------------------------------===//

namespace {
class RetainSummaryManager {

  //==-----------------------------------------------------------------==//
  //  Typedefs.
  //==-----------------------------------------------------------------==//

  typedef llvm::DenseMap<const FunctionDecl*, const RetainSummary *>
          FuncSummariesTy;

  typedef ObjCSummaryCache ObjCMethodSummariesTy;

  typedef llvm::FoldingSetNodeWrapper<RetainSummary> CachedSummaryNode;

  //==-----------------------------------------------------------------==//
  //  Data.
  //==-----------------------------------------------------------------==//

  /// Ctx - The ASTContext object for the analyzed ASTs.
  ASTContext &Ctx;

  /// GCEnabled - Records whether or not the analyzed code runs in GC mode.
  const bool GCEnabled;

  /// Records whether or not the analyzed code runs in ARC mode.
  const bool ARCEnabled;

  /// FuncSummaries - A map from FunctionDecls to summaries.
  FuncSummariesTy FuncSummaries;

  /// ObjCClassMethodSummaries - A map from selectors (for instance methods)
  ///  to summaries.
  ObjCMethodSummariesTy ObjCClassMethodSummaries;

  /// ObjCMethodSummaries - A map from selectors to summaries.
  ObjCMethodSummariesTy ObjCMethodSummaries;

  /// BPAlloc - A BumpPtrAllocator used for allocating summaries, ArgEffects,
  ///  and all other data used by the checker.
  llvm::BumpPtrAllocator BPAlloc;

  /// AF - A factory for ArgEffects objects.
  ArgEffects::Factory AF;

  /// ScratchArgs - A holding buffer for construct ArgEffects.
  ArgEffects ScratchArgs; 

  /// ObjCAllocRetE - Default return effect for methods returning Objective-C
  ///  objects.
  RetEffect ObjCAllocRetE;

  /// ObjCInitRetE - Default return effect for init methods returning
  ///   Objective-C objects.
  RetEffect ObjCInitRetE;

  /// SimpleSummaries - Used for uniquing summaries that don't have special
  /// effects.
  llvm::FoldingSet<CachedSummaryNode> SimpleSummaries;

  //==-----------------------------------------------------------------==//
  //  Methods.
  //==-----------------------------------------------------------------==//

  /// getArgEffects - Returns a persistent ArgEffects object based on the
  ///  data in ScratchArgs.
  ArgEffects getArgEffects();

  enum UnaryFuncKind { cfretain, cfrelease, cfmakecollectable };
  
  const RetainSummary *getUnarySummary(const FunctionType* FT,
                                       UnaryFuncKind func);

  const RetainSummary *getCFSummaryCreateRule(const FunctionDecl *FD);
  const RetainSummary *getCFSummaryGetRule(const FunctionDecl *FD);
  const RetainSummary *getCFCreateGetRuleSummary(const FunctionDecl *FD);

  const RetainSummary *getPersistentSummary(const RetainSummary &OldSumm);

  const RetainSummary *getPersistentSummary(RetEffect RetEff,
                                            ArgEffect ReceiverEff = DoNothing,
                                            ArgEffect DefaultEff = MayEscape) {
    RetainSummary Summ(getArgEffects(), RetEff, DefaultEff, ReceiverEff);
    return getPersistentSummary(Summ);
  }

  const RetainSummary *getDoNothingSummary() {
    return getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, DoNothing);
  }
  
  const RetainSummary *getDefaultSummary() {
    return getPersistentSummary(RetEffect::MakeNoRet(),
                                DoNothing, MayEscape);
  }

  const RetainSummary *getPersistentStopSummary() {
    return getPersistentSummary(RetEffect::MakeNoRet(),
                                StopTracking, StopTracking);
  }

  void InitializeClassMethodSummaries();
  void InitializeMethodSummaries();
private:
  void addNSObjectClsMethSummary(Selector S, const RetainSummary *Summ) {
    ObjCClassMethodSummaries[S] = Summ;
  }

  void addNSObjectMethSummary(Selector S, const RetainSummary *Summ) {
    ObjCMethodSummaries[S] = Summ;
  }

  void addClassMethSummary(const char* Cls, const char* name,
                           const RetainSummary *Summ, bool isNullary = true) {
    IdentifierInfo* ClsII = &Ctx.Idents.get(Cls);
    Selector S = isNullary ? GetNullarySelector(name, Ctx) 
                           : GetUnarySelector(name, Ctx);
    ObjCClassMethodSummaries[ObjCSummaryKey(ClsII, S)]  = Summ;
  }

  void addInstMethSummary(const char* Cls, const char* nullaryName,
                          const RetainSummary *Summ) {
    IdentifierInfo* ClsII = &Ctx.Idents.get(Cls);
    Selector S = GetNullarySelector(nullaryName, Ctx);
    ObjCMethodSummaries[ObjCSummaryKey(ClsII, S)]  = Summ;
  }

  Selector generateSelector(va_list argp) {
    SmallVector<IdentifierInfo*, 10> II;

    while (const char* s = va_arg(argp, const char*))
      II.push_back(&Ctx.Idents.get(s));

    return Ctx.Selectors.getSelector(II.size(), &II[0]);
  }

  void addMethodSummary(IdentifierInfo *ClsII, ObjCMethodSummariesTy& Summaries,
                        const RetainSummary * Summ, va_list argp) {
    Selector S = generateSelector(argp);
    Summaries[ObjCSummaryKey(ClsII, S)] = Summ;
  }

  void addInstMethSummary(const char* Cls, const RetainSummary * Summ, ...) {
    va_list argp;
    va_start(argp, Summ);
    addMethodSummary(&Ctx.Idents.get(Cls), ObjCMethodSummaries, Summ, argp);
    va_end(argp);
  }

  void addClsMethSummary(const char* Cls, const RetainSummary * Summ, ...) {
    va_list argp;
    va_start(argp, Summ);
    addMethodSummary(&Ctx.Idents.get(Cls),ObjCClassMethodSummaries, Summ, argp);
    va_end(argp);
  }

  void addClsMethSummary(IdentifierInfo *II, const RetainSummary * Summ, ...) {
    va_list argp;
    va_start(argp, Summ);
    addMethodSummary(II, ObjCClassMethodSummaries, Summ, argp);
    va_end(argp);
  }

public:

  RetainSummaryManager(ASTContext &ctx, bool gcenabled, bool usesARC)
   : Ctx(ctx),
     GCEnabled(gcenabled),
     ARCEnabled(usesARC),
     AF(BPAlloc), ScratchArgs(AF.getEmptyMap()),
     ObjCAllocRetE(gcenabled
                    ? RetEffect::MakeGCNotOwned()
                    : (usesARC ? RetEffect::MakeARCNotOwned()
                               : RetEffect::MakeOwned(RetEffect::ObjC, true))),
     ObjCInitRetE(gcenabled 
                    ? RetEffect::MakeGCNotOwned()
                    : (usesARC ? RetEffect::MakeARCNotOwned()
                               : RetEffect::MakeOwnedWhenTrackedReceiver())) {
    InitializeClassMethodSummaries();
    InitializeMethodSummaries();
  }

  const RetainSummary *getSummary(const CallEvent &Call,
                                  ProgramStateRef State = 0);

  const RetainSummary *getFunctionSummary(const FunctionDecl *FD);

  const RetainSummary *getMethodSummary(Selector S, const ObjCInterfaceDecl *ID,
                                        const ObjCMethodDecl *MD,
                                        QualType RetTy,
                                        ObjCMethodSummariesTy &CachedSummaries);

  const RetainSummary *getInstanceMethodSummary(const ObjCMethodCall &M,
                                                ProgramStateRef State);

  const RetainSummary *getClassMethodSummary(const ObjCMethodCall &M) {
    assert(!M.isInstanceMessage());
    const ObjCInterfaceDecl *Class = M.getReceiverInterface();

    return getMethodSummary(M.getSelector(), Class, M.getDecl(),
                            M.getResultType(), ObjCClassMethodSummaries);
  }

  /// getMethodSummary - This version of getMethodSummary is used to query
  ///  the summary for the current method being analyzed.
  const RetainSummary *getMethodSummary(const ObjCMethodDecl *MD) {
    const ObjCInterfaceDecl *ID = MD->getClassInterface();
    Selector S = MD->getSelector();
    QualType ResultTy = MD->getResultType();

    ObjCMethodSummariesTy *CachedSummaries;
    if (MD->isInstanceMethod())
      CachedSummaries = &ObjCMethodSummaries;
    else
      CachedSummaries = &ObjCClassMethodSummaries;

    return getMethodSummary(S, ID, MD, ResultTy, *CachedSummaries);
  }

  const RetainSummary *getStandardMethodSummary(const ObjCMethodDecl *MD,
                                                Selector S, QualType RetTy);

  void updateSummaryFromAnnotations(const RetainSummary *&Summ,
                                    const ObjCMethodDecl *MD);

  void updateSummaryFromAnnotations(const RetainSummary *&Summ,
                                    const FunctionDecl *FD);

  void updateSummaryForCall(const RetainSummary *&Summ,
                            const CallEvent &Call);

  bool isGCEnabled() const { return GCEnabled; }

  bool isARCEnabled() const { return ARCEnabled; }
  
  bool isARCorGCEnabled() const { return GCEnabled || ARCEnabled; }

  RetEffect getObjAllocRetEffect() const { return ObjCAllocRetE; }

  friend class RetainSummaryTemplate;
};

// Used to avoid allocating long-term (BPAlloc'd) memory for default retain
// summaries. If a function or method looks like it has a default summary, but
// it has annotations, the annotations are added to the stack-based template
// and then copied into managed memory.
class RetainSummaryTemplate {
  RetainSummaryManager &Manager;
  const RetainSummary *&RealSummary;
  RetainSummary ScratchSummary;
  bool Accessed;
public:
  RetainSummaryTemplate(const RetainSummary *&real, RetainSummaryManager &mgr)
    : Manager(mgr), RealSummary(real), ScratchSummary(*real), Accessed(false) {}

  ~RetainSummaryTemplate() {
    if (Accessed)
      RealSummary = Manager.getPersistentSummary(ScratchSummary);
  }

  RetainSummary &operator*() {
    Accessed = true;
    return ScratchSummary;
  }

  RetainSummary *operator->() {
    Accessed = true;
    return &ScratchSummary;
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Implementation of checker data structures.
//===----------------------------------------------------------------------===//

ArgEffects RetainSummaryManager::getArgEffects() {
  ArgEffects AE = ScratchArgs;
  ScratchArgs = AF.getEmptyMap();
  return AE;
}

const RetainSummary *
RetainSummaryManager::getPersistentSummary(const RetainSummary &OldSumm) {
  // Unique "simple" summaries -- those without ArgEffects.
  if (OldSumm.isSimple()) {
    llvm::FoldingSetNodeID ID;
    OldSumm.Profile(ID);

    void *Pos;
    CachedSummaryNode *N = SimpleSummaries.FindNodeOrInsertPos(ID, Pos);

    if (!N) {
      N = (CachedSummaryNode *) BPAlloc.Allocate<CachedSummaryNode>();
      new (N) CachedSummaryNode(OldSumm);
      SimpleSummaries.InsertNode(N, Pos);
    }

    return &N->getValue();
  }

  RetainSummary *Summ = (RetainSummary *) BPAlloc.Allocate<RetainSummary>();
  new (Summ) RetainSummary(OldSumm);
  return Summ;
}

//===----------------------------------------------------------------------===//
// Summary creation for functions (largely uses of Core Foundation).
//===----------------------------------------------------------------------===//

static bool isRetain(const FunctionDecl *FD, StringRef FName) {
  return FName.endswith("Retain");
}

static bool isRelease(const FunctionDecl *FD, StringRef FName) {
  return FName.endswith("Release");
}

static bool isMakeCollectable(const FunctionDecl *FD, StringRef FName) {
  // FIXME: Remove FunctionDecl parameter.
  // FIXME: Is it really okay if MakeCollectable isn't a suffix?
  return FName.find("MakeCollectable") != StringRef::npos;
}

static ArgEffect getStopTrackingHardEquivalent(ArgEffect E) {
  switch (E) {
  case DoNothing:
  case Autorelease:
  case DecRefBridgedTransfered:
  case IncRef:
  case IncRefMsg:
  case MakeCollectable:
  case MayEscape:
  case NewAutoreleasePool:
  case StopTracking:
  case StopTrackingHard:
    return StopTrackingHard;
  case DecRef:
  case DecRefAndStopTrackingHard:
    return DecRefAndStopTrackingHard;
  case DecRefMsg:
  case DecRefMsgAndStopTrackingHard:
    return DecRefMsgAndStopTrackingHard;
  case Dealloc:
    return Dealloc;
  }

  llvm_unreachable("Unknown ArgEffect kind");
}

void RetainSummaryManager::updateSummaryForCall(const RetainSummary *&S,
                                                const CallEvent &Call) {
  if (Call.hasNonZeroCallbackArg()) {
    ArgEffect RecEffect =
      getStopTrackingHardEquivalent(S->getReceiverEffect());
    ArgEffect DefEffect =
      getStopTrackingHardEquivalent(S->getDefaultArgEffect());

    ArgEffects CustomArgEffects = S->getArgEffects();
    for (ArgEffects::iterator I = CustomArgEffects.begin(),
                              E = CustomArgEffects.end();
         I != E; ++I) {
      ArgEffect Translated = getStopTrackingHardEquivalent(I->second);
      if (Translated != DefEffect)
        ScratchArgs = AF.add(ScratchArgs, I->first, Translated);
    }

    RetEffect RE = RetEffect::MakeNoRetHard();

    // Special cases where the callback argument CANNOT free the return value.
    // This can generally only happen if we know that the callback will only be
    // called when the return value is already being deallocated.
    if (const FunctionCall *FC = dyn_cast<FunctionCall>(&Call)) {
      if (IdentifierInfo *Name = FC->getDecl()->getIdentifier()) {
        // When the CGBitmapContext is deallocated, the callback here will free
        // the associated data buffer.
        if (Name->isStr("CGBitmapContextCreateWithData"))
          RE = S->getRetEffect();
      }
    }

    S = getPersistentSummary(RE, RecEffect, DefEffect);
  }

  // Special case '[super init];' and '[self init];'
  //
  // Even though calling '[super init]' without assigning the result to self
  // and checking if the parent returns 'nil' is a bad pattern, it is common.
  // Additionally, our Self Init checker already warns about it. To avoid
  // overwhelming the user with messages from both checkers, we model the case
  // of '[super init]' in cases when it is not consumed by another expression
  // as if the call preserves the value of 'self'; essentially, assuming it can 
  // never fail and return 'nil'.
  // Note, we don't want to just stop tracking the value since we want the
  // RetainCount checker to report leaks and use-after-free if SelfInit checker
  // is turned off.
  if (const ObjCMethodCall *MC = dyn_cast<ObjCMethodCall>(&Call)) {
    if (MC->getMethodFamily() == OMF_init && MC->isReceiverSelfOrSuper()) {

      // Check if the message is not consumed, we know it will not be used in
      // an assignment, ex: "self = [super init]".
      const Expr *ME = MC->getOriginExpr();
      const LocationContext *LCtx = MC->getLocationContext();
      ParentMap &PM = LCtx->getAnalysisDeclContext()->getParentMap();
      if (!PM.isConsumedExpr(ME)) {
        RetainSummaryTemplate ModifiableSummaryTemplate(S, *this);
        ModifiableSummaryTemplate->setReceiverEffect(DoNothing);
        ModifiableSummaryTemplate->setRetEffect(RetEffect::MakeNoRet());
      }
    }

  }
}

const RetainSummary *
RetainSummaryManager::getSummary(const CallEvent &Call,
                                 ProgramStateRef State) {
  const RetainSummary *Summ;
  switch (Call.getKind()) {
  case CE_Function:
    Summ = getFunctionSummary(cast<FunctionCall>(Call).getDecl());
    break;
  case CE_CXXMember:
  case CE_CXXMemberOperator:
  case CE_Block:
  case CE_CXXConstructor:
  case CE_CXXDestructor:
  case CE_CXXAllocator:
    // FIXME: These calls are currently unsupported.
    return getPersistentStopSummary();
  case CE_ObjCMessage: {
    const ObjCMethodCall &Msg = cast<ObjCMethodCall>(Call);
    if (Msg.isInstanceMessage())
      Summ = getInstanceMethodSummary(Msg, State);
    else
      Summ = getClassMethodSummary(Msg);
    break;
  }
  }

  updateSummaryForCall(Summ, Call);

  assert(Summ && "Unknown call type?");
  return Summ;
}

const RetainSummary *
RetainSummaryManager::getFunctionSummary(const FunctionDecl *FD) {
  // If we don't know what function we're calling, use our default summary.
  if (!FD)
    return getDefaultSummary();

  // Look up a summary in our cache of FunctionDecls -> Summaries.
  FuncSummariesTy::iterator I = FuncSummaries.find(FD);
  if (I != FuncSummaries.end())
    return I->second;

  // No summary?  Generate one.
  const RetainSummary *S = 0;
  bool AllowAnnotations = true;

  do {
    // We generate "stop" summaries for implicitly defined functions.
    if (FD->isImplicit()) {
      S = getPersistentStopSummary();
      break;
    }

    // [PR 3337] Use 'getAs<FunctionType>' to strip away any typedefs on the
    // function's type.
    const FunctionType* FT = FD->getType()->getAs<FunctionType>();
    const IdentifierInfo *II = FD->getIdentifier();
    if (!II)
      break;

    StringRef FName = II->getName();

    // Strip away preceding '_'.  Doing this here will effect all the checks
    // down below.
    FName = FName.substr(FName.find_first_not_of('_'));

    // Inspect the result type.
    QualType RetTy = FT->getResultType();

    // FIXME: This should all be refactored into a chain of "summary lookup"
    //  filters.
    assert(ScratchArgs.isEmpty());

    if (FName == "pthread_create" || FName == "pthread_setspecific") {
      // Part of: <rdar://problem/7299394> and <rdar://problem/11282706>.
      // This will be addressed better with IPA.
      S = getPersistentStopSummary();
    } else if (FName == "NSMakeCollectable") {
      // Handle: id NSMakeCollectable(CFTypeRef)
      S = (RetTy->isObjCIdType())
          ? getUnarySummary(FT, cfmakecollectable)
          : getPersistentStopSummary();
      // The headers on OS X 10.8 use cf_consumed/ns_returns_retained,
      // but we can fully model NSMakeCollectable ourselves.
      AllowAnnotations = false;
    } else if (FName == "CFPlugInInstanceCreate") {
      S = getPersistentSummary(RetEffect::MakeNoRet());
    } else if (FName == "IOBSDNameMatching" ||
               FName == "IOServiceMatching" ||
               FName == "IOServiceNameMatching" ||
               FName == "IORegistryEntrySearchCFProperty" ||
               FName == "IORegistryEntryIDMatching" ||
               FName == "IOOpenFirmwarePathMatching") {
      // Part of <rdar://problem/6961230>. (IOKit)
      // This should be addressed using a API table.
      S = getPersistentSummary(RetEffect::MakeOwned(RetEffect::CF, true),
                               DoNothing, DoNothing);
    } else if (FName == "IOServiceGetMatchingService" ||
               FName == "IOServiceGetMatchingServices") {
      // FIXES: <rdar://problem/6326900>
      // This should be addressed using a API table.  This strcmp is also
      // a little gross, but there is no need to super optimize here.
      ScratchArgs = AF.add(ScratchArgs, 1, DecRef);
      S = getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, DoNothing);
    } else if (FName == "IOServiceAddNotification" ||
               FName == "IOServiceAddMatchingNotification") {
      // Part of <rdar://problem/6961230>. (IOKit)
      // This should be addressed using a API table.
      ScratchArgs = AF.add(ScratchArgs, 2, DecRef);
      S = getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, DoNothing);
    } else if (FName == "CVPixelBufferCreateWithBytes") {
      // FIXES: <rdar://problem/7283567>
      // Eventually this can be improved by recognizing that the pixel
      // buffer passed to CVPixelBufferCreateWithBytes is released via
      // a callback and doing full IPA to make sure this is done correctly.
      // FIXME: This function has an out parameter that returns an
      // allocated object.
      ScratchArgs = AF.add(ScratchArgs, 7, StopTracking);
      S = getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, DoNothing);
    } else if (FName == "CGBitmapContextCreateWithData") {
      // FIXES: <rdar://problem/7358899>
      // Eventually this can be improved by recognizing that 'releaseInfo'
      // passed to CGBitmapContextCreateWithData is released via
      // a callback and doing full IPA to make sure this is done correctly.
      ScratchArgs = AF.add(ScratchArgs, 8, StopTracking);
      S = getPersistentSummary(RetEffect::MakeOwned(RetEffect::CF, true),
                               DoNothing, DoNothing);
    } else if (FName == "CVPixelBufferCreateWithPlanarBytes") {
      // FIXES: <rdar://problem/7283567>
      // Eventually this can be improved by recognizing that the pixel
      // buffer passed to CVPixelBufferCreateWithPlanarBytes is released
      // via a callback and doing full IPA to make sure this is done
      // correctly.
      ScratchArgs = AF.add(ScratchArgs, 12, StopTracking);
      S = getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, DoNothing);
    } else if (FName == "dispatch_set_context") {
      // <rdar://problem/11059275> - The analyzer currently doesn't have
      // a good way to reason about the finalizer function for libdispatch.
      // If we pass a context object that is memory managed, stop tracking it.
      // FIXME: this hack should possibly go away once we can handle
      // libdispatch finalizers.
      ScratchArgs = AF.add(ScratchArgs, 1, StopTracking);
      S = getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, DoNothing);
    } else if (FName.startswith("NSLog")) {
      S = getDoNothingSummary();
    } else if (FName.startswith("NS") &&
                (FName.find("Insert") != StringRef::npos)) {
      // Whitelist NSXXInsertXX, for example NSMapInsertIfAbsent, since they can
      // be deallocated by NSMapRemove. (radar://11152419)
      ScratchArgs = AF.add(ScratchArgs, 1, StopTracking);
      ScratchArgs = AF.add(ScratchArgs, 2, StopTracking);
      S = getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, DoNothing);
    }

    // Did we get a summary?
    if (S)
      break;

    if (RetTy->isPointerType()) {
      if (FD->getAttr<CFAuditedTransferAttr>()) {
        S = getCFCreateGetRuleSummary(FD);
        break;
      }
      
      // For CoreFoundation ('CF') types.
      if (cocoa::isRefType(RetTy, "CF", FName)) {
        if (isRetain(FD, FName))
          S = getUnarySummary(FT, cfretain);
        else if (isMakeCollectable(FD, FName))
          S = getUnarySummary(FT, cfmakecollectable);
        else
          S = getCFCreateGetRuleSummary(FD);

        break;
      }

      // For CoreGraphics ('CG') types.
      if (cocoa::isRefType(RetTy, "CG", FName)) {
        if (isRetain(FD, FName))
          S = getUnarySummary(FT, cfretain);
        else
          S = getCFCreateGetRuleSummary(FD);

        break;
      }

      // For the Disk Arbitration API (DiskArbitration/DADisk.h)
      if (cocoa::isRefType(RetTy, "DADisk") ||
          cocoa::isRefType(RetTy, "DADissenter") ||
          cocoa::isRefType(RetTy, "DASessionRef")) {
        S = getCFCreateGetRuleSummary(FD);
        break;
      }

      break;
    }

    // Check for release functions, the only kind of functions that we care
    // about that don't return a pointer type.
    if (FName[0] == 'C' && (FName[1] == 'F' || FName[1] == 'G')) {
      // Test for 'CGCF'.
      FName = FName.substr(FName.startswith("CGCF") ? 4 : 2);

      if (isRelease(FD, FName))
        S = getUnarySummary(FT, cfrelease);
      else {
        assert (ScratchArgs.isEmpty());
        // Remaining CoreFoundation and CoreGraphics functions.
        // We use to assume that they all strictly followed the ownership idiom
        // and that ownership cannot be transferred.  While this is technically
        // correct, many methods allow a tracked object to escape.  For example:
        //
        //   CFMutableDictionaryRef x = CFDictionaryCreateMutable(...);
        //   CFDictionaryAddValue(y, key, x);
        //   CFRelease(x);
        //   ... it is okay to use 'x' since 'y' has a reference to it
        //
        // We handle this and similar cases with the follow heuristic.  If the
        // function name contains "InsertValue", "SetValue", "AddValue",
        // "AppendValue", or "SetAttribute", then we assume that arguments may
        // "escape."  This means that something else holds on to the object,
        // allowing it be used even after its local retain count drops to 0.
        ArgEffect E = (StrInStrNoCase(FName, "InsertValue") != StringRef::npos||
                       StrInStrNoCase(FName, "AddValue") != StringRef::npos ||
                       StrInStrNoCase(FName, "SetValue") != StringRef::npos ||
                       StrInStrNoCase(FName, "AppendValue") != StringRef::npos||
                       StrInStrNoCase(FName, "SetAttribute") != StringRef::npos)
                      ? MayEscape : DoNothing;

        S = getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, E);
      }
    }
  }
  while (0);

  // If we got all the way here without any luck, use a default summary.
  if (!S)
    S = getDefaultSummary();

  // Annotations override defaults.
  if (AllowAnnotations)
    updateSummaryFromAnnotations(S, FD);

  FuncSummaries[FD] = S;
  return S;
}

const RetainSummary *
RetainSummaryManager::getCFCreateGetRuleSummary(const FunctionDecl *FD) {
  if (coreFoundation::followsCreateRule(FD))
    return getCFSummaryCreateRule(FD);

  return getCFSummaryGetRule(FD);
}

const RetainSummary *
RetainSummaryManager::getUnarySummary(const FunctionType* FT,
                                      UnaryFuncKind func) {

  // Sanity check that this is *really* a unary function.  This can
  // happen if people do weird things.
  const FunctionProtoType* FTP = dyn_cast<FunctionProtoType>(FT);
  if (!FTP || FTP->getNumArgs() != 1)
    return getPersistentStopSummary();

  assert (ScratchArgs.isEmpty());

  ArgEffect Effect;
  switch (func) {
    case cfretain: Effect = IncRef; break;
    case cfrelease: Effect = DecRef; break;
    case cfmakecollectable: Effect = MakeCollectable; break;
  }

  ScratchArgs = AF.add(ScratchArgs, 0, Effect);
  return getPersistentSummary(RetEffect::MakeNoRet(), DoNothing, DoNothing);
}

const RetainSummary * 
RetainSummaryManager::getCFSummaryCreateRule(const FunctionDecl *FD) {
  assert (ScratchArgs.isEmpty());

  return getPersistentSummary(RetEffect::MakeOwned(RetEffect::CF, true));
}

const RetainSummary * 
RetainSummaryManager::getCFSummaryGetRule(const FunctionDecl *FD) {
  assert (ScratchArgs.isEmpty());
  return getPersistentSummary(RetEffect::MakeNotOwned(RetEffect::CF),
                              DoNothing, DoNothing);
}

//===----------------------------------------------------------------------===//
// Summary creation for Selectors.
//===----------------------------------------------------------------------===//

void
RetainSummaryManager::updateSummaryFromAnnotations(const RetainSummary *&Summ,
                                                   const FunctionDecl *FD) {
  if (!FD)
    return;

  assert(Summ && "Must have a summary to add annotations to.");
  RetainSummaryTemplate Template(Summ, *this);

  // Effects on the parameters.
  unsigned parm_idx = 0;
  for (FunctionDecl::param_const_iterator pi = FD->param_begin(), 
         pe = FD->param_end(); pi != pe; ++pi, ++parm_idx) {
    const ParmVarDecl *pd = *pi;
    if (pd->getAttr<NSConsumedAttr>()) {
      if (!GCEnabled) {
        Template->addArg(AF, parm_idx, DecRef);      
      }
    } else if (pd->getAttr<CFConsumedAttr>()) {
      Template->addArg(AF, parm_idx, DecRef);      
    }   
  }
  
  QualType RetTy = FD->getResultType();

  // Determine if there is a special return effect for this method.
  if (cocoa::isCocoaObjectRef(RetTy)) {
    if (FD->getAttr<NSReturnsRetainedAttr>()) {
      Template->setRetEffect(ObjCAllocRetE);
    }
    else if (FD->getAttr<CFReturnsRetainedAttr>()) {
      Template->setRetEffect(RetEffect::MakeOwned(RetEffect::CF, true));
    }
    else if (FD->getAttr<NSReturnsNotRetainedAttr>()) {
      Template->setRetEffect(RetEffect::MakeNotOwned(RetEffect::ObjC));
    }
    else if (FD->getAttr<CFReturnsNotRetainedAttr>()) {
      Template->setRetEffect(RetEffect::MakeNotOwned(RetEffect::CF));
    }
  } else if (RetTy->getAs<PointerType>()) {
    if (FD->getAttr<CFReturnsRetainedAttr>()) {
      Template->setRetEffect(RetEffect::MakeOwned(RetEffect::CF, true));
    }
    else if (FD->getAttr<CFReturnsNotRetainedAttr>()) {
      Template->setRetEffect(RetEffect::MakeNotOwned(RetEffect::CF));
    }
  }
}

void
RetainSummaryManager::updateSummaryFromAnnotations(const RetainSummary *&Summ,
                                                   const ObjCMethodDecl *MD) {
  if (!MD)
    return;

  assert(Summ && "Must have a valid summary to add annotations to");
  RetainSummaryTemplate Template(Summ, *this);
  bool isTrackedLoc = false;

  // Effects on the receiver.
  if (MD->getAttr<NSConsumesSelfAttr>()) {
    if (!GCEnabled)
      Template->setReceiverEffect(DecRefMsg);      
  }
  
  // Effects on the parameters.
  unsigned parm_idx = 0;
  for (ObjCMethodDecl::param_const_iterator
         pi=MD->param_begin(), pe=MD->param_end();
       pi != pe; ++pi, ++parm_idx) {
    const ParmVarDecl *pd = *pi;
    if (pd->getAttr<NSConsumedAttr>()) {
      if (!GCEnabled)
        Template->addArg(AF, parm_idx, DecRef);      
    }
    else if(pd->getAttr<CFConsumedAttr>()) {
      Template->addArg(AF, parm_idx, DecRef);      
    }   
  }
  
  // Determine if there is a special return effect for this method.
  if (cocoa::isCocoaObjectRef(MD->getResultType())) {
    if (MD->getAttr<NSReturnsRetainedAttr>()) {
      Template->setRetEffect(ObjCAllocRetE);
      return;
    }
    if (MD->getAttr<NSReturnsNotRetainedAttr>()) {
      Template->setRetEffect(RetEffect::MakeNotOwned(RetEffect::ObjC));
      return;
    }

    isTrackedLoc = true;
  } else {
    isTrackedLoc = MD->getResultType()->getAs<PointerType>() != NULL;
  }

  if (isTrackedLoc) {
    if (MD->getAttr<CFReturnsRetainedAttr>())
      Template->setRetEffect(RetEffect::MakeOwned(RetEffect::CF, true));
    else if (MD->getAttr<CFReturnsNotRetainedAttr>())
      Template->setRetEffect(RetEffect::MakeNotOwned(RetEffect::CF));
  }
}

const RetainSummary *
RetainSummaryManager::getStandardMethodSummary(const ObjCMethodDecl *MD,
                                               Selector S, QualType RetTy) {
  // Any special effects?
  ArgEffect ReceiverEff = DoNothing;
  RetEffect ResultEff = RetEffect::MakeNoRet();

  // Check the method family, and apply any default annotations.
  switch (MD ? MD->getMethodFamily() : S.getMethodFamily()) {
    case OMF_None:
    case OMF_performSelector:
      // Assume all Objective-C methods follow Cocoa Memory Management rules.
      // FIXME: Does the non-threaded performSelector family really belong here?
      // The selector could be, say, @selector(copy).
      if (cocoa::isCocoaObjectRef(RetTy))
        ResultEff = RetEffect::MakeNotOwned(RetEffect::ObjC);
      else if (coreFoundation::isCFObjectRef(RetTy)) {
        // ObjCMethodDecl currently doesn't consider CF objects as valid return 
        // values for alloc, new, copy, or mutableCopy, so we have to
        // double-check with the selector. This is ugly, but there aren't that
        // many Objective-C methods that return CF objects, right?
        if (MD) {
          switch (S.getMethodFamily()) {
          case OMF_alloc:
          case OMF_new:
          case OMF_copy:
          case OMF_mutableCopy:
            ResultEff = RetEffect::MakeOwned(RetEffect::CF, true);
            break;
          default:
            ResultEff = RetEffect::MakeNotOwned(RetEffect::CF);        
            break;
          }
        } else {
          ResultEff = RetEffect::MakeNotOwned(RetEffect::CF);        
        }
      }
      break;
    case OMF_init:
      ResultEff = ObjCInitRetE;
      ReceiverEff = DecRefMsg;
      break;
    case OMF_alloc:
    case OMF_new:
    case OMF_copy:
    case OMF_mutableCopy:
      if (cocoa::isCocoaObjectRef(RetTy))
        ResultEff = ObjCAllocRetE;
      else if (coreFoundation::isCFObjectRef(RetTy))
        ResultEff = RetEffect::MakeOwned(RetEffect::CF, true);
      break;
    case OMF_autorelease:
      ReceiverEff = Autorelease;
      break;
    case OMF_retain:
      ReceiverEff = IncRefMsg;
      break;
    case OMF_release:
      ReceiverEff = DecRefMsg;
      break;
    case OMF_dealloc:
      ReceiverEff = Dealloc;
      break;
    case OMF_self:
      // -self is handled specially by the ExprEngine to propagate the receiver.
      break;
    case OMF_retainCount:
    case OMF_finalize:
      // These methods don't return objects.
      break;
  }

  // If one of the arguments in the selector has the keyword 'delegate' we
  // should stop tracking the reference count for the receiver.  This is
  // because the reference count is quite possibly handled by a delegate
  // method.
  if (S.isKeywordSelector()) {
    for (unsigned i = 0, e = S.getNumArgs(); i != e; ++i) {
      StringRef Slot = S.getNameForSlot(i);
      if (Slot.substr(Slot.size() - 8).equals_lower("delegate")) {
        if (ResultEff == ObjCInitRetE)
          ResultEff = RetEffect::MakeNoRetHard();
        else
          ReceiverEff = StopTrackingHard;
      }
    }
  }

  if (ScratchArgs.isEmpty() && ReceiverEff == DoNothing &&
      ResultEff.getKind() == RetEffect::NoRet)
    return getDefaultSummary();

  return getPersistentSummary(ResultEff, ReceiverEff, MayEscape);
}

const RetainSummary *
RetainSummaryManager::getInstanceMethodSummary(const ObjCMethodCall &Msg,
                                               ProgramStateRef State) {
  const ObjCInterfaceDecl *ReceiverClass = 0;

  // We do better tracking of the type of the object than the core ExprEngine.
  // See if we have its type in our private state.
  // FIXME: Eventually replace the use of state->get<RefBindings> with
  // a generic API for reasoning about the Objective-C types of symbolic
  // objects.
  SVal ReceiverV = Msg.getReceiverSVal();
  if (SymbolRef Sym = ReceiverV.getAsLocSymbol())
    if (const RefVal *T = getRefBinding(State, Sym))
      if (const ObjCObjectPointerType *PT =
            T->getType()->getAs<ObjCObjectPointerType>())
        ReceiverClass = PT->getInterfaceDecl();

  // If we don't know what kind of object this is, fall back to its static type.
  if (!ReceiverClass)
    ReceiverClass = Msg.getReceiverInterface();

  // FIXME: The receiver could be a reference to a class, meaning that
  //  we should use the class method.
  // id x = [NSObject class];
  // [x performSelector:... withObject:... afterDelay:...];
  Selector S = Msg.getSelector();
  const ObjCMethodDecl *Method = Msg.getDecl();
  if (!Method && ReceiverClass)
    Method = ReceiverClass->getInstanceMethod(S);

  return getMethodSummary(S, ReceiverClass, Method, Msg.getResultType(),
                          ObjCMethodSummaries);
}

const RetainSummary *
RetainSummaryManager::getMethodSummary(Selector S, const ObjCInterfaceDecl *ID,
                                       const ObjCMethodDecl *MD, QualType RetTy,
                                       ObjCMethodSummariesTy &CachedSummaries) {

  // Look up a summary in our summary cache.
  const RetainSummary *Summ = CachedSummaries.find(ID, S);

  if (!Summ) {
    Summ = getStandardMethodSummary(MD, S, RetTy);

    // Annotations override defaults.
    updateSummaryFromAnnotations(Summ, MD);

    // Memoize the summary.
    CachedSummaries[ObjCSummaryKey(ID, S)] = Summ;
  }

  return Summ;
}

void RetainSummaryManager::InitializeClassMethodSummaries() {
  assert(ScratchArgs.isEmpty());
  // Create the [NSAssertionHandler currentHander] summary.
  addClassMethSummary("NSAssertionHandler", "currentHandler",
                getPersistentSummary(RetEffect::MakeNotOwned(RetEffect::ObjC)));

  // Create the [NSAutoreleasePool addObject:] summary.
  ScratchArgs = AF.add(ScratchArgs, 0, Autorelease);
  addClassMethSummary("NSAutoreleasePool", "addObject",
                      getPersistentSummary(RetEffect::MakeNoRet(),
                                           DoNothing, Autorelease));
}

void RetainSummaryManager::InitializeMethodSummaries() {

  assert (ScratchArgs.isEmpty());

  // Create the "init" selector.  It just acts as a pass-through for the
  // receiver.
  const RetainSummary *InitSumm = getPersistentSummary(ObjCInitRetE, DecRefMsg);
  addNSObjectMethSummary(GetNullarySelector("init", Ctx), InitSumm);

  // awakeAfterUsingCoder: behaves basically like an 'init' method.  It
  // claims the receiver and returns a retained object.
  addNSObjectMethSummary(GetUnarySelector("awakeAfterUsingCoder", Ctx),
                         InitSumm);

  // The next methods are allocators.
  const RetainSummary *AllocSumm = getPersistentSummary(ObjCAllocRetE);
  const RetainSummary *CFAllocSumm =
    getPersistentSummary(RetEffect::MakeOwned(RetEffect::CF, true));

  // Create the "retain" selector.
  RetEffect NoRet = RetEffect::MakeNoRet();
  const RetainSummary *Summ = getPersistentSummary(NoRet, IncRefMsg);
  addNSObjectMethSummary(GetNullarySelector("retain", Ctx), Summ);

  // Create the "release" selector.
  Summ = getPersistentSummary(NoRet, DecRefMsg);
  addNSObjectMethSummary(GetNullarySelector("release", Ctx), Summ);

  // Create the "drain" selector.
  Summ = getPersistentSummary(NoRet, isGCEnabled() ? DoNothing : DecRef);
  addNSObjectMethSummary(GetNullarySelector("drain", Ctx), Summ);

  // Create the -dealloc summary.
  Summ = getPersistentSummary(NoRet, Dealloc);
  addNSObjectMethSummary(GetNullarySelector("dealloc", Ctx), Summ);

  // Create the "autorelease" selector.
  Summ = getPersistentSummary(NoRet, Autorelease);
  addNSObjectMethSummary(GetNullarySelector("autorelease", Ctx), Summ);

  // Specially handle NSAutoreleasePool.
  addInstMethSummary("NSAutoreleasePool", "init",
                     getPersistentSummary(NoRet, NewAutoreleasePool));

  // For NSWindow, allocated objects are (initially) self-owned.
  // FIXME: For now we opt for false negatives with NSWindow, as these objects
  //  self-own themselves.  However, they only do this once they are displayed.
  //  Thus, we need to track an NSWindow's display status.
  //  This is tracked in <rdar://problem/6062711>.
  //  See also http://llvm.org/bugs/show_bug.cgi?id=3714.
  const RetainSummary *NoTrackYet = getPersistentSummary(RetEffect::MakeNoRet(),
                                                   StopTracking,
                                                   StopTracking);

  addClassMethSummary("NSWindow", "alloc", NoTrackYet);

  // For NSPanel (which subclasses NSWindow), allocated objects are not
  //  self-owned.
  // FIXME: For now we don't track NSPanels. object for the same reason
  //   as for NSWindow objects.
  addClassMethSummary("NSPanel", "alloc", NoTrackYet);

  // Don't track allocated autorelease pools yet, as it is okay to prematurely
  // exit a method.
  addClassMethSummary("NSAutoreleasePool", "alloc", NoTrackYet);
  addClassMethSummary("NSAutoreleasePool", "allocWithZone", NoTrackYet, false);

  // Create summaries QCRenderer/QCView -createSnapShotImageOfType:
  addInstMethSummary("QCRenderer", AllocSumm,
                     "createSnapshotImageOfType", NULL);
  addInstMethSummary("QCView", AllocSumm,
                     "createSnapshotImageOfType", NULL);

  // Create summaries for CIContext, 'createCGImage' and
  // 'createCGLayerWithSize'.  These objects are CF objects, and are not
  // automatically garbage collected.
  addInstMethSummary("CIContext", CFAllocSumm,
                     "createCGImage", "fromRect", NULL);
  addInstMethSummary("CIContext", CFAllocSumm,
                     "createCGImage", "fromRect", "format", "colorSpace", NULL);
  addInstMethSummary("CIContext", CFAllocSumm, "createCGLayerWithSize",
           "info", NULL);
}

//===----------------------------------------------------------------------===//
// Error reporting.
//===----------------------------------------------------------------------===//
namespace {
  typedef llvm::DenseMap<const ExplodedNode *, const RetainSummary *>
    SummaryLogTy;

  //===-------------===//
  // Bug Descriptions. //
  //===-------------===//

  class CFRefBug : public BugType {
  protected:
    CFRefBug(StringRef name)
    : BugType(name, categories::MemoryCoreFoundationObjectiveC) {}
  public:

    // FIXME: Eventually remove.
    virtual const char *getDescription() const = 0;

    virtual bool isLeak() const { return false; }
  };

  class UseAfterRelease : public CFRefBug {
  public:
    UseAfterRelease() : CFRefBug("Use-after-release") {}

    const char *getDescription() const {
      return "Reference-counted object is used after it is released";
    }
  };

  class BadRelease : public CFRefBug {
  public:
    BadRelease() : CFRefBug("Bad release") {}

    const char *getDescription() const {
      return "Incorrect decrement of the reference count of an object that is "
             "not owned at this point by the caller";
    }
  };

  class DeallocGC : public CFRefBug {
  public:
    DeallocGC()
    : CFRefBug("-dealloc called while using garbage collection") {}

    const char *getDescription() const {
      return "-dealloc called while using garbage collection";
    }
  };

  class DeallocNotOwned : public CFRefBug {
  public:
    DeallocNotOwned()
    : CFRefBug("-dealloc sent to non-exclusively owned object") {}

    const char *getDescription() const {
      return "-dealloc sent to object that may be referenced elsewhere";
    }
  };

  class OverAutorelease : public CFRefBug {
  public:
    OverAutorelease()
    : CFRefBug("Object sent -autorelease too many times") {}

    const char *getDescription() const {
      return "Object sent -autorelease too many times";
    }
  };

  class ReturnedNotOwnedForOwned : public CFRefBug {
  public:
    ReturnedNotOwnedForOwned()
    : CFRefBug("Method should return an owned object") {}

    const char *getDescription() const {
      return "Object with a +0 retain count returned to caller where a +1 "
             "(owning) retain count is expected";
    }
  };

  class Leak : public CFRefBug {
  public:
    Leak(StringRef name)
    : CFRefBug(name) {
      // Leaks should not be reported if they are post-dominated by a sink.
      setSuppressOnSink(true);
    }

    const char *getDescription() const { return ""; }

    bool isLeak() const { return true; }
  };

  //===---------===//
  // Bug Reports.  //
  //===---------===//

  class CFRefReportVisitor : public BugReporterVisitorImpl<CFRefReportVisitor> {
  protected:
    SymbolRef Sym;
    const SummaryLogTy &SummaryLog;
    bool GCEnabled;
    
  public:
    CFRefReportVisitor(SymbolRef sym, bool gcEnabled, const SummaryLogTy &log)
       : Sym(sym), SummaryLog(log), GCEnabled(gcEnabled) {}

    virtual void Profile(llvm::FoldingSetNodeID &ID) const {
      static int x = 0;
      ID.AddPointer(&x);
      ID.AddPointer(Sym);
    }

    virtual PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                           const ExplodedNode *PrevN,
                                           BugReporterContext &BRC,
                                           BugReport &BR);

    virtual PathDiagnosticPiece *getEndPath(BugReporterContext &BRC,
                                            const ExplodedNode *N,
                                            BugReport &BR);
  };

  class CFRefLeakReportVisitor : public CFRefReportVisitor {
  public:
    CFRefLeakReportVisitor(SymbolRef sym, bool GCEnabled,
                           const SummaryLogTy &log)
       : CFRefReportVisitor(sym, GCEnabled, log) {}

    PathDiagnosticPiece *getEndPath(BugReporterContext &BRC,
                                    const ExplodedNode *N,
                                    BugReport &BR);

    virtual BugReporterVisitor *clone() const {
      // The curiously-recurring template pattern only works for one level of
      // subclassing. Rather than make a new template base for
      // CFRefReportVisitor, we simply override clone() to do the right thing.
      // This could be trouble someday if BugReporterVisitorImpl is ever
      // used for something else besides a convenient implementation of clone().
      return new CFRefLeakReportVisitor(*this);
    }
  };

  class CFRefReport : public BugReport {
    void addGCModeDescription(const LangOptions &LOpts, bool GCEnabled);

  public:
    CFRefReport(CFRefBug &D, const LangOptions &LOpts, bool GCEnabled,
                const SummaryLogTy &Log, ExplodedNode *n, SymbolRef sym,
                bool registerVisitor = true)
      : BugReport(D, D.getDescription(), n) {
      if (registerVisitor)
        addVisitor(new CFRefReportVisitor(sym, GCEnabled, Log));
      addGCModeDescription(LOpts, GCEnabled);
    }

    CFRefReport(CFRefBug &D, const LangOptions &LOpts, bool GCEnabled,
                const SummaryLogTy &Log, ExplodedNode *n, SymbolRef sym,
                StringRef endText)
      : BugReport(D, D.getDescription(), endText, n) {
      addVisitor(new CFRefReportVisitor(sym, GCEnabled, Log));
      addGCModeDescription(LOpts, GCEnabled);
    }

    virtual std::pair<ranges_iterator, ranges_iterator> getRanges() {
      const CFRefBug& BugTy = static_cast<CFRefBug&>(getBugType());
      if (!BugTy.isLeak())
        return BugReport::getRanges();
      else
        return std::make_pair(ranges_iterator(), ranges_iterator());
    }
  };

  class CFRefLeakReport : public CFRefReport {
    const MemRegion* AllocBinding;

  public:
    CFRefLeakReport(CFRefBug &D, const LangOptions &LOpts, bool GCEnabled,
                    const SummaryLogTy &Log, ExplodedNode *n, SymbolRef sym,
                    CheckerContext &Ctx);

    PathDiagnosticLocation getLocation(const SourceManager &SM) const {
      assert(Location.isValid());
      return Location;
    }
  };
} // end anonymous namespace

void CFRefReport::addGCModeDescription(const LangOptions &LOpts,
                                       bool GCEnabled) {
  const char *GCModeDescription = 0;

  switch (LOpts.getGC()) {
  case LangOptions::GCOnly:
    assert(GCEnabled);
    GCModeDescription = "Code is compiled to only use garbage collection";
    break;

  case LangOptions::NonGC:
    assert(!GCEnabled);
    GCModeDescription = "Code is compiled to use reference counts";
    break;

  case LangOptions::HybridGC:
    if (GCEnabled) {
      GCModeDescription = "Code is compiled to use either garbage collection "
                          "(GC) or reference counts (non-GC).  The bug occurs "
                          "with GC enabled";
      break;
    } else {
      GCModeDescription = "Code is compiled to use either garbage collection "
                          "(GC) or reference counts (non-GC).  The bug occurs "
                          "in non-GC mode";
      break;
    }
  }

  assert(GCModeDescription && "invalid/unknown GC mode");
  addExtraText(GCModeDescription);
}

// FIXME: This should be a method on SmallVector.
static inline bool contains(const SmallVectorImpl<ArgEffect>& V,
                            ArgEffect X) {
  for (SmallVectorImpl<ArgEffect>::const_iterator I=V.begin(), E=V.end();
       I!=E; ++I)
    if (*I == X) return true;

  return false;
}

static bool isNumericLiteralExpression(const Expr *E) {
  // FIXME: This set of cases was copied from SemaExprObjC.
  return isa<IntegerLiteral>(E) || 
         isa<CharacterLiteral>(E) ||
         isa<FloatingLiteral>(E) ||
         isa<ObjCBoolLiteralExpr>(E) ||
         isa<CXXBoolLiteralExpr>(E);
}

PathDiagnosticPiece *CFRefReportVisitor::VisitNode(const ExplodedNode *N,
                                                   const ExplodedNode *PrevN,
                                                   BugReporterContext &BRC,
                                                   BugReport &BR) {
  // FIXME: We will eventually need to handle non-statement-based events
  // (__attribute__((cleanup))).
  if (!isa<StmtPoint>(N->getLocation()))
    return NULL;

  // Check if the type state has changed.
  ProgramStateRef PrevSt = PrevN->getState();
  ProgramStateRef CurrSt = N->getState();
  const LocationContext *LCtx = N->getLocationContext();

  const RefVal* CurrT = getRefBinding(CurrSt, Sym);
  if (!CurrT) return NULL;

  const RefVal &CurrV = *CurrT;
  const RefVal *PrevT = getRefBinding(PrevSt, Sym);

  // Create a string buffer to constain all the useful things we want
  // to tell the user.
  std::string sbuf;
  llvm::raw_string_ostream os(sbuf);

  // This is the allocation site since the previous node had no bindings
  // for this symbol.
  if (!PrevT) {
    const Stmt *S = cast<StmtPoint>(N->getLocation()).getStmt();

    if (isa<ObjCArrayLiteral>(S)) {
      os << "NSArray literal is an object with a +0 retain count";
    }
    else if (isa<ObjCDictionaryLiteral>(S)) {
      os << "NSDictionary literal is an object with a +0 retain count";
    }
    else if (const ObjCBoxedExpr *BL = dyn_cast<ObjCBoxedExpr>(S)) {
      if (isNumericLiteralExpression(BL->getSubExpr()))
        os << "NSNumber literal is an object with a +0 retain count";
      else {
        const ObjCInterfaceDecl *BoxClass = 0;
        if (const ObjCMethodDecl *Method = BL->getBoxingMethod())
          BoxClass = Method->getClassInterface();

        // We should always be able to find the boxing class interface,
        // but consider this future-proofing.
        if (BoxClass)
          os << *BoxClass << " b";
        else
          os << "B";

        os << "oxed expression produces an object with a +0 retain count";
      }
    }
    else {      
      if (const CallExpr *CE = dyn_cast<CallExpr>(S)) {
        // Get the name of the callee (if it is available).
        SVal X = CurrSt->getSValAsScalarOrLoc(CE->getCallee(), LCtx);
        if (const FunctionDecl *FD = X.getAsFunctionDecl())
          os << "Call to function '" << *FD << '\'';
        else
          os << "function call";
      }
      else {
        assert(isa<ObjCMessageExpr>(S));
        CallEventManager &Mgr = CurrSt->getStateManager().getCallEventManager();
        CallEventRef<ObjCMethodCall> Call
          = Mgr.getObjCMethodCall(cast<ObjCMessageExpr>(S), CurrSt, LCtx);

        switch (Call->getMessageKind()) {
        case OCM_Message:
          os << "Method";
          break;
        case OCM_PropertyAccess:
          os << "Property";
          break;
        case OCM_Subscript:
          os << "Subscript";
          break;
        }
      }

      if (CurrV.getObjKind() == RetEffect::CF) {
        os << " returns a Core Foundation object with a ";
      }
      else {
        assert (CurrV.getObjKind() == RetEffect::ObjC);
        os << " returns an Objective-C object with a ";
      }

      if (CurrV.isOwned()) {
        os << "+1 retain count";

        if (GCEnabled) {
          assert(CurrV.getObjKind() == RetEffect::CF);
          os << ".  "
          "Core Foundation objects are not automatically garbage collected.";
        }
      }
      else {
        assert (CurrV.isNotOwned());
        os << "+0 retain count";
      }
    }

    PathDiagnosticLocation Pos(S, BRC.getSourceManager(),
                                  N->getLocationContext());
    return new PathDiagnosticEventPiece(Pos, os.str());
  }

  // Gather up the effects that were performed on the object at this
  // program point
  SmallVector<ArgEffect, 2> AEffects;

  const ExplodedNode *OrigNode = BRC.getNodeResolver().getOriginalNode(N);
  if (const RetainSummary *Summ = SummaryLog.lookup(OrigNode)) {
    // We only have summaries attached to nodes after evaluating CallExpr and
    // ObjCMessageExprs.
    const Stmt *S = cast<StmtPoint>(N->getLocation()).getStmt();

    if (const CallExpr *CE = dyn_cast<CallExpr>(S)) {
      // Iterate through the parameter expressions and see if the symbol
      // was ever passed as an argument.
      unsigned i = 0;

      for (CallExpr::const_arg_iterator AI=CE->arg_begin(), AE=CE->arg_end();
           AI!=AE; ++AI, ++i) {

        // Retrieve the value of the argument.  Is it the symbol
        // we are interested in?
        if (CurrSt->getSValAsScalarOrLoc(*AI, LCtx).getAsLocSymbol() != Sym)
          continue;

        // We have an argument.  Get the effect!
        AEffects.push_back(Summ->getArg(i));
      }
    }
    else if (const ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(S)) {
      if (const Expr *receiver = ME->getInstanceReceiver())
        if (CurrSt->getSValAsScalarOrLoc(receiver, LCtx)
              .getAsLocSymbol() == Sym) {
          // The symbol we are tracking is the receiver.
          AEffects.push_back(Summ->getReceiverEffect());
        }
    }
  }

  do {
    // Get the previous type state.
    RefVal PrevV = *PrevT;

    // Specially handle -dealloc.
    if (!GCEnabled && contains(AEffects, Dealloc)) {
      // Determine if the object's reference count was pushed to zero.
      assert(!(PrevV == CurrV) && "The typestate *must* have changed.");
      // We may not have transitioned to 'release' if we hit an error.
      // This case is handled elsewhere.
      if (CurrV.getKind() == RefVal::Released) {
        assert(CurrV.getCombinedCounts() == 0);
        os << "Object released by directly sending the '-dealloc' message";
        break;
      }
    }

    // Specially handle CFMakeCollectable and friends.
    if (contains(AEffects, MakeCollectable)) {
      // Get the name of the function.
      const Stmt *S = cast<StmtPoint>(N->getLocation()).getStmt();
      SVal X =
        CurrSt->getSValAsScalarOrLoc(cast<CallExpr>(S)->getCallee(), LCtx);
      const FunctionDecl *FD = X.getAsFunctionDecl();

      if (GCEnabled) {
        // Determine if the object's reference count was pushed to zero.
        assert(!(PrevV == CurrV) && "The typestate *must* have changed.");

        os << "In GC mode a call to '" << *FD
        <<  "' decrements an object's retain count and registers the "
        "object with the garbage collector. ";

        if (CurrV.getKind() == RefVal::Released) {
          assert(CurrV.getCount() == 0);
          os << "Since it now has a 0 retain count the object can be "
          "automatically collected by the garbage collector.";
        }
        else
          os << "An object must have a 0 retain count to be garbage collected. "
          "After this call its retain count is +" << CurrV.getCount()
          << '.';
      }
      else
        os << "When GC is not enabled a call to '" << *FD
        << "' has no effect on its argument.";

      // Nothing more to say.
      break;
    }

    // Determine if the typestate has changed.
    if (!(PrevV == CurrV))
      switch (CurrV.getKind()) {
        case RefVal::Owned:
        case RefVal::NotOwned:

          if (PrevV.getCount() == CurrV.getCount()) {
            // Did an autorelease message get sent?
            if (PrevV.getAutoreleaseCount() == CurrV.getAutoreleaseCount())
              return 0;

            assert(PrevV.getAutoreleaseCount() < CurrV.getAutoreleaseCount());
            os << "Object sent -autorelease message";
            break;
          }

          if (PrevV.getCount() > CurrV.getCount())
            os << "Reference count decremented.";
          else
            os << "Reference count incremented.";

          if (unsigned Count = CurrV.getCount())
            os << " The object now has a +" << Count << " retain count.";

          if (PrevV.getKind() == RefVal::Released) {
            assert(GCEnabled && CurrV.getCount() > 0);
            os << " The object is not eligible for garbage collection until "
                  "the retain count reaches 0 again.";
          }

          break;

        case RefVal::Released:
          os << "Object released.";
          break;

        case RefVal::ReturnedOwned:
          // Autoreleases can be applied after marking a node ReturnedOwned.
          if (CurrV.getAutoreleaseCount())
            return NULL;

          os << "Object returned to caller as an owning reference (single "
                "retain count transferred to caller)";
          break;

        case RefVal::ReturnedNotOwned:
          os << "Object returned to caller with a +0 retain count";
          break;

        default:
          return NULL;
      }

    // Emit any remaining diagnostics for the argument effects (if any).
    for (SmallVectorImpl<ArgEffect>::iterator I=AEffects.begin(),
         E=AEffects.end(); I != E; ++I) {

      // A bunch of things have alternate behavior under GC.
      if (GCEnabled)
        switch (*I) {
          default: break;
          case Autorelease:
            os << "In GC mode an 'autorelease' has no effect.";
            continue;
          case IncRefMsg:
            os << "In GC mode the 'retain' message has no effect.";
            continue;
          case DecRefMsg:
            os << "In GC mode the 'release' message has no effect.";
            continue;
        }
    }
  } while (0);

  if (os.str().empty())
    return 0; // We have nothing to say!

  const Stmt *S = cast<StmtPoint>(N->getLocation()).getStmt();
  PathDiagnosticLocation Pos(S, BRC.getSourceManager(),
                                N->getLocationContext());
  PathDiagnosticPiece *P = new PathDiagnosticEventPiece(Pos, os.str());

  // Add the range by scanning the children of the statement for any bindings
  // to Sym.
  for (Stmt::const_child_iterator I = S->child_begin(), E = S->child_end();
       I!=E; ++I)
    if (const Expr *Exp = dyn_cast_or_null<Expr>(*I))
      if (CurrSt->getSValAsScalarOrLoc(Exp, LCtx).getAsLocSymbol() == Sym) {
        P->addRange(Exp->getSourceRange());
        break;
      }

  return P;
}

// Find the first node in the current function context that referred to the
// tracked symbol and the memory location that value was stored to. Note, the
// value is only reported if the allocation occurred in the same function as
// the leak.
static std::pair<const ExplodedNode*,const MemRegion*>
GetAllocationSite(ProgramStateManager& StateMgr, const ExplodedNode *N,
                  SymbolRef Sym) {
  const ExplodedNode *Last = N;
  const MemRegion* FirstBinding = 0;
  const LocationContext *LeakContext = N->getLocationContext();

  while (N) {
    ProgramStateRef St = N->getState();

    if (!getRefBinding(St, Sym))
      break;

    StoreManager::FindUniqueBinding FB(Sym);
    StateMgr.iterBindings(St, FB);
    if (FB) FirstBinding = FB.getRegion();

    // Allocation node, is the last node in the current context in which the
    // symbol was tracked.
    if (N->getLocationContext() == LeakContext)
      Last = N;

    N = N->pred_empty() ? NULL : *(N->pred_begin());
  }

  // If allocation happened in a function different from the leak node context,
  // do not report the binding.
  assert(N && "Could not find allocation node");
  if (N->getLocationContext() != LeakContext) {
    FirstBinding = 0;
  }

  return std::make_pair(Last, FirstBinding);
}

PathDiagnosticPiece*
CFRefReportVisitor::getEndPath(BugReporterContext &BRC,
                               const ExplodedNode *EndN,
                               BugReport &BR) {
  BR.markInteresting(Sym);
  return BugReporterVisitor::getDefaultEndPath(BRC, EndN, BR);
}

PathDiagnosticPiece*
CFRefLeakReportVisitor::getEndPath(BugReporterContext &BRC,
                                   const ExplodedNode *EndN,
                                   BugReport &BR) {

  // Tell the BugReporterContext to report cases when the tracked symbol is
  // assigned to different variables, etc.
  BR.markInteresting(Sym);

  // We are reporting a leak.  Walk up the graph to get to the first node where
  // the symbol appeared, and also get the first VarDecl that tracked object
  // is stored to.
  const ExplodedNode *AllocNode = 0;
  const MemRegion* FirstBinding = 0;

  llvm::tie(AllocNode, FirstBinding) =
    GetAllocationSite(BRC.getStateManager(), EndN, Sym);

  SourceManager& SM = BRC.getSourceManager();

  // Compute an actual location for the leak.  Sometimes a leak doesn't
  // occur at an actual statement (e.g., transition between blocks; end
  // of function) so we need to walk the graph and compute a real location.
  const ExplodedNode *LeakN = EndN;
  PathDiagnosticLocation L = PathDiagnosticLocation::createEndOfPath(LeakN, SM);

  std::string sbuf;
  llvm::raw_string_ostream os(sbuf);

  os << "Object leaked: ";

  if (FirstBinding) {
    os << "object allocated and stored into '"
       << FirstBinding->getString() << '\'';
  }
  else
    os << "allocated object";

  // Get the retain count.
  const RefVal* RV = getRefBinding(EndN->getState(), Sym);
  assert(RV);

  if (RV->getKind() == RefVal::ErrorLeakReturned) {
    // FIXME: Per comments in rdar://6320065, "create" only applies to CF
    // objects.  Only "copy", "alloc", "retain" and "new" transfer ownership
    // to the caller for NS objects.
    const Decl *D = &EndN->getCodeDecl();
    
    os << (isa<ObjCMethodDecl>(D) ? " is returned from a method "
                                  : " is returned from a function ");
    
    if (D->getAttr<CFReturnsNotRetainedAttr>())
      os << "that is annotated as CF_RETURNS_NOT_RETAINED";
    else if (D->getAttr<NSReturnsNotRetainedAttr>())
      os << "that is annotated as NS_RETURNS_NOT_RETAINED";
    else {
      if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
        os << "whose name ('" << MD->getSelector().getAsString()
           << "') does not start with 'copy', 'mutableCopy', 'alloc' or 'new'."
              "  This violates the naming convention rules"
              " given in the Memory Management Guide for Cocoa";
      }
      else {
        const FunctionDecl *FD = cast<FunctionDecl>(D);
        os << "whose name ('" << *FD
           << "') does not contain 'Copy' or 'Create'.  This violates the naming"
              " convention rules given in the Memory Management Guide for Core"
              " Foundation";
      }
    }
  }
  else if (RV->getKind() == RefVal::ErrorGCLeakReturned) {
    ObjCMethodDecl &MD = cast<ObjCMethodDecl>(EndN->getCodeDecl());
    os << " and returned from method '" << MD.getSelector().getAsString()
       << "' is potentially leaked when using garbage collection.  Callers "
          "of this method do not expect a returned object with a +1 retain "
          "count since they expect the object to be managed by the garbage "
          "collector";
  }
  else
    os << " is not referenced later in this execution path and has a retain "
          "count of +" << RV->getCount();

  return new PathDiagnosticEventPiece(L, os.str());
}

CFRefLeakReport::CFRefLeakReport(CFRefBug &D, const LangOptions &LOpts,
                                 bool GCEnabled, const SummaryLogTy &Log, 
                                 ExplodedNode *n, SymbolRef sym,
                                 CheckerContext &Ctx)
: CFRefReport(D, LOpts, GCEnabled, Log, n, sym, false) {

  // Most bug reports are cached at the location where they occurred.
  // With leaks, we want to unique them by the location where they were
  // allocated, and only report a single path.  To do this, we need to find
  // the allocation site of a piece of tracked memory, which we do via a
  // call to GetAllocationSite.  This will walk the ExplodedGraph backwards.
  // Note that this is *not* the trimmed graph; we are guaranteed, however,
  // that all ancestor nodes that represent the allocation site have the
  // same SourceLocation.
  const ExplodedNode *AllocNode = 0;

  const SourceManager& SMgr = Ctx.getSourceManager();

  llvm::tie(AllocNode, AllocBinding) =  // Set AllocBinding.
    GetAllocationSite(Ctx.getStateManager(), getErrorNode(), sym);

  // Get the SourceLocation for the allocation site.
  // FIXME: This will crash the analyzer if an allocation comes from an
  // implicit call. (Currently there are no such allocations in Cocoa, though.)
  const Stmt *AllocStmt;
  ProgramPoint P = AllocNode->getLocation();
  if (CallExitEnd *Exit = dyn_cast<CallExitEnd>(&P))
    AllocStmt = Exit->getCalleeContext()->getCallSite();
  else
    AllocStmt = cast<PostStmt>(P).getStmt();
  assert(AllocStmt && "All allocations must come from explicit calls");
  Location = PathDiagnosticLocation::createBegin(AllocStmt, SMgr,
                                                  n->getLocationContext());
  // Fill in the description of the bug.
  Description.clear();
  llvm::raw_string_ostream os(Description);
  os << "Potential leak ";
  if (GCEnabled)
    os << "(when using garbage collection) ";
  os << "of an object";

  // FIXME: AllocBinding doesn't get populated for RegionStore yet.
  if (AllocBinding)
    os << " stored into '" << AllocBinding->getString() << '\'';

  addVisitor(new CFRefLeakReportVisitor(sym, GCEnabled, Log));
}

//===----------------------------------------------------------------------===//
// Main checker logic.
//===----------------------------------------------------------------------===//

namespace {
class RetainCountChecker
  : public Checker< check::Bind,
                    check::DeadSymbols,
                    check::EndAnalysis,
                    check::EndPath,
                    check::PostStmt<BlockExpr>,
                    check::PostStmt<CastExpr>,
                    check::PostStmt<ObjCArrayLiteral>,
                    check::PostStmt<ObjCDictionaryLiteral>,
                    check::PostStmt<ObjCBoxedExpr>,
                    check::PostCall,
                    check::PreStmt<ReturnStmt>,
                    check::RegionChanges,
                    eval::Assume,
                    eval::Call > {
  mutable OwningPtr<CFRefBug> useAfterRelease, releaseNotOwned;
  mutable OwningPtr<CFRefBug> deallocGC, deallocNotOwned;
  mutable OwningPtr<CFRefBug> overAutorelease, returnNotOwnedForOwned;
  mutable OwningPtr<CFRefBug> leakWithinFunction, leakAtReturn;
  mutable OwningPtr<CFRefBug> leakWithinFunctionGC, leakAtReturnGC;

  typedef llvm::DenseMap<SymbolRef, const SimpleProgramPointTag *> SymbolTagMap;

  // This map is only used to ensure proper deletion of any allocated tags.
  mutable SymbolTagMap DeadSymbolTags;

  mutable OwningPtr<RetainSummaryManager> Summaries;
  mutable OwningPtr<RetainSummaryManager> SummariesGC;
  mutable SummaryLogTy SummaryLog;
  mutable bool ShouldResetSummaryLog;

public:  
  RetainCountChecker() : ShouldResetSummaryLog(false) {}

  virtual ~RetainCountChecker() {
    DeleteContainerSeconds(DeadSymbolTags);
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

  CFRefBug *getLeakWithinFunctionBug(const LangOptions &LOpts,
                                     bool GCEnabled) const {
    if (GCEnabled) {
      if (!leakWithinFunctionGC)
        leakWithinFunctionGC.reset(new Leak("Leak of object when using "
                                             "garbage collection"));
      return leakWithinFunctionGC.get();
    } else {
      if (!leakWithinFunction) {
        if (LOpts.getGC() == LangOptions::HybridGC) {
          leakWithinFunction.reset(new Leak("Leak of object when not using "
                                            "garbage collection (GC) in "
                                            "dual GC/non-GC code"));
        } else {
          leakWithinFunction.reset(new Leak("Leak"));
        }
      }
      return leakWithinFunction.get();
    }
  }

  CFRefBug *getLeakAtReturnBug(const LangOptions &LOpts, bool GCEnabled) const {
    if (GCEnabled) {
      if (!leakAtReturnGC)
        leakAtReturnGC.reset(new Leak("Leak of returned object when using "
                                      "garbage collection"));
      return leakAtReturnGC.get();
    } else {
      if (!leakAtReturn) {
        if (LOpts.getGC() == LangOptions::HybridGC) {
          leakAtReturn.reset(new Leak("Leak of returned object when not using "
                                      "garbage collection (GC) in dual "
                                      "GC/non-GC code"));
        } else {
          leakAtReturn.reset(new Leak("Leak of returned object"));
        }
      }
      return leakAtReturn.get();
    }
  }

  RetainSummaryManager &getSummaryManager(ASTContext &Ctx,
                                          bool GCEnabled) const {
    // FIXME: We don't support ARC being turned on and off during one analysis.
    // (nor, for that matter, do we support changing ASTContexts)
    bool ARCEnabled = (bool)Ctx.getLangOpts().ObjCAutoRefCount;
    if (GCEnabled) {
      if (!SummariesGC)
        SummariesGC.reset(new RetainSummaryManager(Ctx, true, ARCEnabled));
      else
        assert(SummariesGC->isARCEnabled() == ARCEnabled);
      return *SummariesGC;
    } else {
      if (!Summaries)
        Summaries.reset(new RetainSummaryManager(Ctx, false, ARCEnabled));
      else
        assert(Summaries->isARCEnabled() == ARCEnabled);
      return *Summaries;
    }
  }

  RetainSummaryManager &getSummaryManager(CheckerContext &C) const {
    return getSummaryManager(C.getASTContext(), C.isObjCGCEnabled());
  }

  void printState(raw_ostream &Out, ProgramStateRef State,
                  const char *NL, const char *Sep) const;

  void checkBind(SVal loc, SVal val, const Stmt *S, CheckerContext &C) const;
  void checkPostStmt(const BlockExpr *BE, CheckerContext &C) const;
  void checkPostStmt(const CastExpr *CE, CheckerContext &C) const;

  void checkPostStmt(const ObjCArrayLiteral *AL, CheckerContext &C) const;
  void checkPostStmt(const ObjCDictionaryLiteral *DL, CheckerContext &C) const;
  void checkPostStmt(const ObjCBoxedExpr *BE, CheckerContext &C) const;

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
                     const StoreManager::InvalidatedSymbols *invalidated,
                     ArrayRef<const MemRegion *> ExplicitRegions,
                     ArrayRef<const MemRegion *> Regions,
                     const CallEvent *Call) const;
                                        
  bool wantsRegionChangeUpdate(ProgramStateRef state) const {
    return true;
  }

  void checkPreStmt(const ReturnStmt *S, CheckerContext &C) const;
  void checkReturnWithRetEffect(const ReturnStmt *S, CheckerContext &C,
                                ExplodedNode *Pred, RetEffect RE, RefVal X,
                                SymbolRef Sym, ProgramStateRef state) const;
                                              
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
  void checkEndPath(CheckerContext &C) const;

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

  std::pair<ExplodedNode *, ProgramStateRef >
  handleAutoreleaseCounts(ProgramStateRef state, ExplodedNode *Pred,
                          const ProgramPointTag *Tag, CheckerContext &Ctx,
                          SymbolRef Sym, RefVal V) const;

  ExplodedNode *processLeaks(ProgramStateRef state,
                             SmallVectorImpl<SymbolRef> &Leaked,
                             CheckerContext &Ctx,
                             ExplodedNode *Pred = 0) const;
};
} // end anonymous namespace

namespace {
class StopTrackingCallback : public SymbolVisitor {
  ProgramStateRef state;
public:
  StopTrackingCallback(ProgramStateRef st) : state(st) {}
  ProgramStateRef getState() const { return state; }

  bool VisitSymbol(SymbolRef sym) {
    state = state->remove<RefBindings>(sym);
    return true;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Handle statements that may have an effect on refcounts.
//===----------------------------------------------------------------------===//

void RetainCountChecker::checkPostStmt(const BlockExpr *BE,
                                       CheckerContext &C) const {

  // Scan the BlockDecRefExprs for any object the retain count checker
  // may be tracking.
  if (!BE->getBlockDecl()->hasCaptures())
    return;

  ProgramStateRef state = C.getState();
  const BlockDataRegion *R =
    cast<BlockDataRegion>(state->getSVal(BE,
                                         C.getLocationContext()).getAsRegion());

  BlockDataRegion::referenced_vars_iterator I = R->referenced_vars_begin(),
                                            E = R->referenced_vars_end();

  if (I == E)
    return;

  // FIXME: For now we invalidate the tracking of all symbols passed to blocks
  // via captured variables, even though captured variables result in a copy
  // and in implicit increment/decrement of a retain count.
  SmallVector<const MemRegion*, 10> Regions;
  const LocationContext *LC = C.getLocationContext();
  MemRegionManager &MemMgr = C.getSValBuilder().getRegionManager();

  for ( ; I != E; ++I) {
    const VarRegion *VR = *I;
    if (VR->getSuperRegion() == R) {
      VR = MemMgr.getVarRegion(VR->getDecl(), LC);
    }
    Regions.push_back(VR);
  }

  state =
    state->scanReachableSymbols<StopTrackingCallback>(Regions.data(),
                                    Regions.data() + Regions.size()).getState();
  C.addTransition(state);
}

void RetainCountChecker::checkPostStmt(const CastExpr *CE,
                                       CheckerContext &C) const {
  const ObjCBridgedCastExpr *BE = dyn_cast<ObjCBridgedCastExpr>(CE);
  if (!BE)
    return;
  
  ArgEffect AE = IncRef;
  
  switch (BE->getBridgeKind()) {
    case clang::OBC_Bridge:
      // Do nothing.
      return;
    case clang::OBC_BridgeRetained:
      AE = IncRef;
      break;      
    case clang::OBC_BridgeTransfer:
      AE = DecRefBridgedTransfered;
      break;
  }
  
  ProgramStateRef state = C.getState();
  SymbolRef Sym = state->getSVal(CE, C.getLocationContext()).getAsLocSymbol();
  if (!Sym)
    return;
  const RefVal* T = getRefBinding(state, Sym);
  if (!T)
    return;

  RefVal::Kind hasErr = (RefVal::Kind) 0;
  state = updateSymbol(state, Sym, *T, AE, hasErr, C);
  
  if (hasErr) {
    // FIXME: If we get an error during a bridge cast, should we report it?
    // Should we assert that there is no error?
    return;
  }

  C.addTransition(state);
}

void RetainCountChecker::processObjCLiterals(CheckerContext &C,
                                             const Expr *Ex) const {
  ProgramStateRef state = C.getState();
  const ExplodedNode *pred = C.getPredecessor();  
  for (Stmt::const_child_iterator it = Ex->child_begin(), et = Ex->child_end() ;
       it != et ; ++it) {
    const Stmt *child = *it;
    SVal V = state->getSVal(child, pred->getLocationContext());
    if (SymbolRef sym = V.getAsSymbol())
      if (const RefVal* T = getRefBinding(state, sym)) {
        RefVal::Kind hasErr = (RefVal::Kind) 0;
        state = updateSymbol(state, sym, *T, MayEscape, hasErr, C);
        if (hasErr) {
          processNonLeakError(state, child->getSourceRange(), hasErr, sym, C);
          return;
        }
      }
  }
  
  // Return the object as autoreleased.
  //  RetEffect RE = RetEffect::MakeNotOwned(RetEffect::ObjC);
  if (SymbolRef sym = 
        state->getSVal(Ex, pred->getLocationContext()).getAsSymbol()) {
    QualType ResultTy = Ex->getType();
    state = setRefBinding(state, sym,
                          RefVal::makeNotOwned(RetEffect::ObjC, ResultTy));
  }
  
  C.addTransition(state);  
}

void RetainCountChecker::checkPostStmt(const ObjCArrayLiteral *AL,
                                       CheckerContext &C) const {
  // Apply the 'MayEscape' to all values.
  processObjCLiterals(C, AL);
}

void RetainCountChecker::checkPostStmt(const ObjCDictionaryLiteral *DL,
                                       CheckerContext &C) const {
  // Apply the 'MayEscape' to all keys and values.
  processObjCLiterals(C, DL);
}

void RetainCountChecker::checkPostStmt(const ObjCBoxedExpr *Ex,
                                       CheckerContext &C) const {
  const ExplodedNode *Pred = C.getPredecessor();  
  const LocationContext *LCtx = Pred->getLocationContext();
  ProgramStateRef State = Pred->getState();

  if (SymbolRef Sym = State->getSVal(Ex, LCtx).getAsSymbol()) {
    QualType ResultTy = Ex->getType();
    State = setRefBinding(State, Sym,
                          RefVal::makeNotOwned(RetEffect::ObjC, ResultTy));
  }

  C.addTransition(State);
}

void RetainCountChecker::checkPostCall(const CallEvent &Call,
                                       CheckerContext &C) const {
  RetainSummaryManager &Summaries = getSummaryManager(C);
  const RetainSummary *Summ = Summaries.getSummary(Call, C.getState());

  if (C.wasInlined) {
    processSummaryOfInlined(*Summ, Call, C);
    return;
  }
  checkSummary(*Summ, Call, C);
}

/// GetReturnType - Used to get the return type of a message expression or
///  function call with the intention of affixing that type to a tracked symbol.
///  While the return type can be queried directly from RetEx, when
///  invoking class methods we augment to the return type to be that of
///  a pointer to the class (as opposed it just being id).
// FIXME: We may be able to do this with related result types instead.
// This function is probably overestimating.
static QualType GetReturnType(const Expr *RetE, ASTContext &Ctx) {
  QualType RetTy = RetE->getType();
  // If RetE is not a message expression just return its type.
  // If RetE is a message expression, return its types if it is something
  /// more specific than id.
  if (const ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(RetE))
    if (const ObjCObjectPointerType *PT = RetTy->getAs<ObjCObjectPointerType>())
      if (PT->isObjCQualifiedIdType() || PT->isObjCIdType() ||
          PT->isObjCClassType()) {
        // At this point we know the return type of the message expression is
        // id, id<...>, or Class. If we have an ObjCInterfaceDecl, we know this
        // is a call to a class method whose type we can resolve.  In such
        // cases, promote the return type to XXX* (where XXX is the class).
        const ObjCInterfaceDecl *D = ME->getReceiverInterface();
        return !D ? RetTy :
                    Ctx.getObjCObjectPointerType(Ctx.getObjCInterfaceType(D));
      }

  return RetTy;
}

// We don't always get the exact modeling of the function with regards to the
// retain count checker even when the function is inlined. For example, we need
// to stop tracking the symbols which were marked with StopTrackingHard.
void RetainCountChecker::processSummaryOfInlined(const RetainSummary &Summ,
                                                 const CallEvent &CallOrMsg,
                                                 CheckerContext &C) const {
  ProgramStateRef state = C.getState();

  // Evaluate the effect of the arguments.
  for (unsigned idx = 0, e = CallOrMsg.getNumArgs(); idx != e; ++idx) {
    if (Summ.getArg(idx) == StopTrackingHard) {
      SVal V = CallOrMsg.getArgSVal(idx);
      if (SymbolRef Sym = V.getAsLocSymbol()) {
        state = removeRefBinding(state, Sym);
      }
    }
  }

  // Evaluate the effect on the message receiver.
  const ObjCMethodCall *MsgInvocation = dyn_cast<ObjCMethodCall>(&CallOrMsg);
  if (MsgInvocation) {
    if (SymbolRef Sym = MsgInvocation->getReceiverSVal().getAsLocSymbol()) {
      if (Summ.getReceiverEffect() == StopTrackingHard) {
        state = removeRefBinding(state, Sym);
      }
    }
  }

  // Consult the summary for the return value.
  RetEffect RE = Summ.getRetEffect();
  if (RE.getKind() == RetEffect::NoRetHard) {
    SymbolRef Sym = CallOrMsg.getReturnValue().getAsSymbol();
    if (Sym)
      state = removeRefBinding(state, Sym);
  }
  
  C.addTransition(state);
}

void RetainCountChecker::checkSummary(const RetainSummary &Summ,
                                      const CallEvent &CallOrMsg,
                                      CheckerContext &C) const {
  ProgramStateRef state = C.getState();

  // Evaluate the effect of the arguments.
  RefVal::Kind hasErr = (RefVal::Kind) 0;
  SourceRange ErrorRange;
  SymbolRef ErrorSym = 0;

  for (unsigned idx = 0, e = CallOrMsg.getNumArgs(); idx != e; ++idx) {
    SVal V = CallOrMsg.getArgSVal(idx);

    if (SymbolRef Sym = V.getAsLocSymbol()) {
      if (const RefVal *T = getRefBinding(state, Sym)) {
        state = updateSymbol(state, Sym, *T, Summ.getArg(idx), hasErr, C);
        if (hasErr) {
          ErrorRange = CallOrMsg.getArgSourceRange(idx);
          ErrorSym = Sym;
          break;
        }
      }
    }
  }

  // Evaluate the effect on the message receiver.
  bool ReceiverIsTracked = false;
  if (!hasErr) {
    const ObjCMethodCall *MsgInvocation = dyn_cast<ObjCMethodCall>(&CallOrMsg);
    if (MsgInvocation) {
      if (SymbolRef Sym = MsgInvocation->getReceiverSVal().getAsLocSymbol()) {
        if (const RefVal *T = getRefBinding(state, Sym)) {
          ReceiverIsTracked = true;
          state = updateSymbol(state, Sym, *T, Summ.getReceiverEffect(),
                                 hasErr, C);
          if (hasErr) {
            ErrorRange = MsgInvocation->getOriginExpr()->getReceiverRange();
            ErrorSym = Sym;
          }
        }
      }
    }
  }

  // Process any errors.
  if (hasErr) {
    processNonLeakError(state, ErrorRange, hasErr, ErrorSym, C);
    return;
  }

  // Consult the summary for the return value.
  RetEffect RE = Summ.getRetEffect();

  if (RE.getKind() == RetEffect::OwnedWhenTrackedReceiver) {
    if (ReceiverIsTracked)
      RE = getSummaryManager(C).getObjAllocRetEffect();      
    else
      RE = RetEffect::MakeNoRet();
  }

  switch (RE.getKind()) {
    default:
      llvm_unreachable("Unhandled RetEffect.");

    case RetEffect::NoRet:
    case RetEffect::NoRetHard:
      // No work necessary.
      break;

    case RetEffect::OwnedAllocatedSymbol:
    case RetEffect::OwnedSymbol: {
      SymbolRef Sym = CallOrMsg.getReturnValue().getAsSymbol();
      if (!Sym)
        break;

      // Use the result type from the CallEvent as it automatically adjusts
      // for methods/functions that return references.
      QualType ResultTy = CallOrMsg.getResultType();
      state = setRefBinding(state, Sym, RefVal::makeOwned(RE.getObjKind(),
                                                          ResultTy));

      // FIXME: Add a flag to the checker where allocations are assumed to
      // *not* fail.
      break;
    }

    case RetEffect::GCNotOwnedSymbol:
    case RetEffect::ARCNotOwnedSymbol:
    case RetEffect::NotOwnedSymbol: {
      const Expr *Ex = CallOrMsg.getOriginExpr();
      SymbolRef Sym = CallOrMsg.getReturnValue().getAsSymbol();
      if (!Sym)
        break;
      assert(Ex);
      // Use GetReturnType in order to give [NSFoo alloc] the type NSFoo *.
      QualType ResultTy = GetReturnType(Ex, C.getASTContext());
      state = setRefBinding(state, Sym, RefVal::makeNotOwned(RE.getObjKind(),
                                                             ResultTy));
      break;
    }
  }

  // This check is actually necessary; otherwise the statement builder thinks
  // we've hit a previously-found path.
  // Normally addTransition takes care of this, but we want the node pointer.
  ExplodedNode *NewNode;
  if (state == C.getState()) {
    NewNode = C.getPredecessor();
  } else {
    NewNode = C.addTransition(state);
  }

  // Annotate the node with summary we used.
  if (NewNode) {
    // FIXME: This is ugly. See checkEndAnalysis for why it's necessary.
    if (ShouldResetSummaryLog) {
      SummaryLog.clear();
      ShouldResetSummaryLog = false;
    }
    SummaryLog[NewNode] = &Summ;
  }
}


ProgramStateRef 
RetainCountChecker::updateSymbol(ProgramStateRef state, SymbolRef sym,
                                 RefVal V, ArgEffect E, RefVal::Kind &hasErr,
                                 CheckerContext &C) const {
  // In GC mode [... release] and [... retain] do nothing.
  // In ARC mode they shouldn't exist at all, but we just ignore them.
  bool IgnoreRetainMsg = C.isObjCGCEnabled();
  if (!IgnoreRetainMsg)
    IgnoreRetainMsg = (bool)C.getASTContext().getLangOpts().ObjCAutoRefCount;

  switch (E) {
  default:
    break;
  case IncRefMsg:
    E = IgnoreRetainMsg ? DoNothing : IncRef;
    break;
  case DecRefMsg:
    E = IgnoreRetainMsg ? DoNothing : DecRef;
    break;
  case DecRefMsgAndStopTrackingHard:
    E = IgnoreRetainMsg ? StopTracking : DecRefAndStopTrackingHard;
    break;
  case MakeCollectable:
    E = C.isObjCGCEnabled() ? DecRef : DoNothing;
    break;
  case NewAutoreleasePool:
    E = C.isObjCGCEnabled() ? DoNothing : NewAutoreleasePool;
    break;
  }

  // Handle all use-after-releases.
  if (!C.isObjCGCEnabled() && V.getKind() == RefVal::Released) {
    V = V ^ RefVal::ErrorUseAfterRelease;
    hasErr = V.getKind();
    return setRefBinding(state, sym, V);
  }

  switch (E) {
    case DecRefMsg:
    case IncRefMsg:
    case MakeCollectable:
    case DecRefMsgAndStopTrackingHard:
      llvm_unreachable("DecRefMsg/IncRefMsg/MakeCollectable already converted");

    case Dealloc:
      // Any use of -dealloc in GC is *bad*.
      if (C.isObjCGCEnabled()) {
        V = V ^ RefVal::ErrorDeallocGC;
        hasErr = V.getKind();
        break;
      }

      switch (V.getKind()) {
        default:
          llvm_unreachable("Invalid RefVal state for an explicit dealloc.");
        case RefVal::Owned:
          // The object immediately transitions to the released state.
          V = V ^ RefVal::Released;
          V.clearCounts();
          return setRefBinding(state, sym, V);
        case RefVal::NotOwned:
          V = V ^ RefVal::ErrorDeallocNotOwned;
          hasErr = V.getKind();
          break;
      }
      break;

    case NewAutoreleasePool:
      assert(!C.isObjCGCEnabled());
      return state;

    case MayEscape:
      if (V.getKind() == RefVal::Owned) {
        V = V ^ RefVal::NotOwned;
        break;
      }

      // Fall-through.

    case DoNothing:
      return state;

    case Autorelease:
      if (C.isObjCGCEnabled())
        return state;
      // Update the autorelease counts.
      V = V.autorelease();
      break;

    case StopTracking:
    case StopTrackingHard:
      return removeRefBinding(state, sym);

    case IncRef:
      switch (V.getKind()) {
        default:
          llvm_unreachable("Invalid RefVal state for a retain.");
        case RefVal::Owned:
        case RefVal::NotOwned:
          V = V + 1;
          break;
        case RefVal::Released:
          // Non-GC cases are handled above.
          assert(C.isObjCGCEnabled());
          V = (V ^ RefVal::Owned) + 1;
          break;
      }
      break;

    case DecRef:
    case DecRefBridgedTransfered:
    case DecRefAndStopTrackingHard:
      switch (V.getKind()) {
        default:
          // case 'RefVal::Released' handled above.
          llvm_unreachable("Invalid RefVal state for a release.");

        case RefVal::Owned:
          assert(V.getCount() > 0);
          if (V.getCount() == 1)
            V = V ^ (E == DecRefBridgedTransfered ? 
                      RefVal::NotOwned : RefVal::Released);
          else if (E == DecRefAndStopTrackingHard)
            return removeRefBinding(state, sym);

          V = V - 1;
          break;

        case RefVal::NotOwned:
          if (V.getCount() > 0) {
            if (E == DecRefAndStopTrackingHard)
              return removeRefBinding(state, sym);
            V = V - 1;
          } else {
            V = V ^ RefVal::ErrorReleaseNotOwned;
            hasErr = V.getKind();
          }
          break;

        case RefVal::Released:
          // Non-GC cases are handled above.
          assert(C.isObjCGCEnabled());
          V = V ^ RefVal::ErrorUseAfterRelease;
          hasErr = V.getKind();
          break;
      }
      break;
  }
  return setRefBinding(state, sym, V);
}

void RetainCountChecker::processNonLeakError(ProgramStateRef St,
                                             SourceRange ErrorRange,
                                             RefVal::Kind ErrorKind,
                                             SymbolRef Sym,
                                             CheckerContext &C) const {
  ExplodedNode *N = C.generateSink(St);
  if (!N)
    return;

  CFRefBug *BT;
  switch (ErrorKind) {
    default:
      llvm_unreachable("Unhandled error.");
    case RefVal::ErrorUseAfterRelease:
      if (!useAfterRelease)
        useAfterRelease.reset(new UseAfterRelease());
      BT = &*useAfterRelease;
      break;
    case RefVal::ErrorReleaseNotOwned:
      if (!releaseNotOwned)
        releaseNotOwned.reset(new BadRelease());
      BT = &*releaseNotOwned;
      break;
    case RefVal::ErrorDeallocGC:
      if (!deallocGC)
        deallocGC.reset(new DeallocGC());
      BT = &*deallocGC;
      break;
    case RefVal::ErrorDeallocNotOwned:
      if (!deallocNotOwned)
        deallocNotOwned.reset(new DeallocNotOwned());
      BT = &*deallocNotOwned;
      break;
  }

  assert(BT);
  CFRefReport *report = new CFRefReport(*BT, C.getASTContext().getLangOpts(),
                                        C.isObjCGCEnabled(), SummaryLog,
                                        N, Sym);
  report->addRange(ErrorRange);
  C.emitReport(report);
}

//===----------------------------------------------------------------------===//
// Handle the return values of retain-count-related functions.
//===----------------------------------------------------------------------===//

bool RetainCountChecker::evalCall(const CallExpr *CE, CheckerContext &C) const {
  // Get the callee. We're only interested in simple C functions.
  ProgramStateRef state = C.getState();
  const FunctionDecl *FD = C.getCalleeDecl(CE);
  if (!FD)
    return false;

  IdentifierInfo *II = FD->getIdentifier();
  if (!II)
    return false;

  // For now, we're only handling the functions that return aliases of their
  // arguments: CFRetain and CFMakeCollectable (and their families).
  // Eventually we should add other functions we can model entirely,
  // such as CFRelease, which don't invalidate their arguments or globals.
  if (CE->getNumArgs() != 1)
    return false;

  // Get the name of the function.
  StringRef FName = II->getName();
  FName = FName.substr(FName.find_first_not_of('_'));

  // See if it's one of the specific functions we know how to eval.
  bool canEval = false;

  QualType ResultTy = CE->getCallReturnType();
  if (ResultTy->isObjCIdType()) {
    // Handle: id NSMakeCollectable(CFTypeRef)
    canEval = II->isStr("NSMakeCollectable");
  } else if (ResultTy->isPointerType()) {
    // Handle: (CF|CG)Retain
    //         CFMakeCollectable
    // It's okay to be a little sloppy here (CGMakeCollectable doesn't exist).
    if (cocoa::isRefType(ResultTy, "CF", FName) ||
        cocoa::isRefType(ResultTy, "CG", FName)) {
      canEval = isRetain(FD, FName) || isMakeCollectable(FD, FName);
    }
  }
        
  if (!canEval)
    return false;

  // Bind the return value.
  const LocationContext *LCtx = C.getLocationContext();
  SVal RetVal = state->getSVal(CE->getArg(0), LCtx);
  if (RetVal.isUnknown()) {
    // If the receiver is unknown, conjure a return value.
    SValBuilder &SVB = C.getSValBuilder();
    RetVal = SVB.conjureSymbolVal(0, CE, LCtx, ResultTy, C.blockCount());
  }
  state = state->BindExpr(CE, LCtx, RetVal, false);

  // FIXME: This should not be necessary, but otherwise the argument seems to be
  // considered alive during the next statement.
  if (const MemRegion *ArgRegion = RetVal.getAsRegion()) {
    // Save the refcount status of the argument.
    SymbolRef Sym = RetVal.getAsLocSymbol();
    const RefVal *Binding = 0;
    if (Sym)
      Binding = getRefBinding(state, Sym);

    // Invalidate the argument region.
    state = state->invalidateRegions(ArgRegion, CE, C.blockCount(), LCtx);

    // Restore the refcount status of the argument.
    if (Binding)
      state = setRefBinding(state, Sym, *Binding);
  }

  C.addTransition(state);
  return true;
}

//===----------------------------------------------------------------------===//
// Handle return statements.
//===----------------------------------------------------------------------===//

void RetainCountChecker::checkPreStmt(const ReturnStmt *S,
                                      CheckerContext &C) const {

  // Only adjust the reference count if this is the top-level call frame,
  // and not the result of inlining.  In the future, we should do
  // better checking even for inlined calls, and see if they match
  // with their expected semantics (e.g., the method should return a retained
  // object, etc.).
  if (!C.inTopFrame())
    return;

  const Expr *RetE = S->getRetValue();
  if (!RetE)
    return;

  ProgramStateRef state = C.getState();
  SymbolRef Sym =
    state->getSValAsScalarOrLoc(RetE, C.getLocationContext()).getAsLocSymbol();
  if (!Sym)
    return;

  // Get the reference count binding (if any).
  const RefVal *T = getRefBinding(state, Sym);
  if (!T)
    return;

  // Change the reference count.
  RefVal X = *T;

  switch (X.getKind()) {
    case RefVal::Owned: {
      unsigned cnt = X.getCount();
      assert(cnt > 0);
      X.setCount(cnt - 1);
      X = X ^ RefVal::ReturnedOwned;
      break;
    }

    case RefVal::NotOwned: {
      unsigned cnt = X.getCount();
      if (cnt) {
        X.setCount(cnt - 1);
        X = X ^ RefVal::ReturnedOwned;
      }
      else {
        X = X ^ RefVal::ReturnedNotOwned;
      }
      break;
    }

    default:
      return;
  }

  // Update the binding.
  state = setRefBinding(state, Sym, X);
  ExplodedNode *Pred = C.addTransition(state);

  // At this point we have updated the state properly.
  // Everything after this is merely checking to see if the return value has
  // been over- or under-retained.

  // Did we cache out?
  if (!Pred)
    return;

  // Update the autorelease counts.
  static SimpleProgramPointTag
         AutoreleaseTag("RetainCountChecker : Autorelease");
  llvm::tie(Pred, state) = handleAutoreleaseCounts(state, Pred, &AutoreleaseTag,
                                                   C, Sym, X);

  // Did we cache out?
  if (!Pred)
    return;

  // Get the updated binding.
  T = getRefBinding(state, Sym);
  assert(T);
  X = *T;

  // Consult the summary of the enclosing method.
  RetainSummaryManager &Summaries = getSummaryManager(C);
  const Decl *CD = &Pred->getCodeDecl();
  RetEffect RE = RetEffect::MakeNoRet();

  // FIXME: What is the convention for blocks? Is there one?
  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(CD)) {
    const RetainSummary *Summ = Summaries.getMethodSummary(MD);
    RE = Summ->getRetEffect();
  } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(CD)) {
    if (!isa<CXXMethodDecl>(FD)) {
      const RetainSummary *Summ = Summaries.getFunctionSummary(FD);
      RE = Summ->getRetEffect();
    }
  }

  checkReturnWithRetEffect(S, C, Pred, RE, X, Sym, state);
}

void RetainCountChecker::checkReturnWithRetEffect(const ReturnStmt *S,
                                                  CheckerContext &C,
                                                  ExplodedNode *Pred,
                                                  RetEffect RE, RefVal X,
                                                  SymbolRef Sym,
                                              ProgramStateRef state) const {
  // Any leaks or other errors?
  if (X.isReturnedOwned() && X.getCount() == 0) {
    if (RE.getKind() != RetEffect::NoRet) {
      bool hasError = false;
      if (C.isObjCGCEnabled() && RE.getObjKind() == RetEffect::ObjC) {
        // Things are more complicated with garbage collection.  If the
        // returned object is suppose to be an Objective-C object, we have
        // a leak (as the caller expects a GC'ed object) because no
        // method should return ownership unless it returns a CF object.
        hasError = true;
        X = X ^ RefVal::ErrorGCLeakReturned;
      }
      else if (!RE.isOwned()) {
        // Either we are using GC and the returned object is a CF type
        // or we aren't using GC.  In either case, we expect that the
        // enclosing method is expected to return ownership.
        hasError = true;
        X = X ^ RefVal::ErrorLeakReturned;
      }

      if (hasError) {
        // Generate an error node.
        state = setRefBinding(state, Sym, X);

        static SimpleProgramPointTag
               ReturnOwnLeakTag("RetainCountChecker : ReturnsOwnLeak");
        ExplodedNode *N = C.addTransition(state, Pred, &ReturnOwnLeakTag);
        if (N) {
          const LangOptions &LOpts = C.getASTContext().getLangOpts();
          bool GCEnabled = C.isObjCGCEnabled();
          CFRefReport *report =
            new CFRefLeakReport(*getLeakAtReturnBug(LOpts, GCEnabled),
                                LOpts, GCEnabled, SummaryLog,
                                N, Sym, C);
          C.emitReport(report);
        }
      }
    }
  } else if (X.isReturnedNotOwned()) {
    if (RE.isOwned()) {
      // Trying to return a not owned object to a caller expecting an
      // owned object.
      state = setRefBinding(state, Sym, X ^ RefVal::ErrorReturnedNotOwned);

      static SimpleProgramPointTag
             ReturnNotOwnedTag("RetainCountChecker : ReturnNotOwnedForOwned");
      ExplodedNode *N = C.addTransition(state, Pred, &ReturnNotOwnedTag);
      if (N) {
        if (!returnNotOwnedForOwned)
          returnNotOwnedForOwned.reset(new ReturnedNotOwnedForOwned());

        CFRefReport *report =
            new CFRefReport(*returnNotOwnedForOwned,
                            C.getASTContext().getLangOpts(), 
                            C.isObjCGCEnabled(), SummaryLog, N, Sym);
        C.emitReport(report);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Check various ways a symbol can be invalidated.
//===----------------------------------------------------------------------===//

void RetainCountChecker::checkBind(SVal loc, SVal val, const Stmt *S,
                                   CheckerContext &C) const {
  // Are we storing to something that causes the value to "escape"?
  bool escapes = true;

  // A value escapes in three possible cases (this may change):
  //
  // (1) we are binding to something that is not a memory region.
  // (2) we are binding to a memregion that does not have stack storage
  // (3) we are binding to a memregion with stack storage that the store
  //     does not understand.
  ProgramStateRef state = C.getState();

  if (loc::MemRegionVal *regionLoc = dyn_cast<loc::MemRegionVal>(&loc)) {
    escapes = !regionLoc->getRegion()->hasStackStorage();

    if (!escapes) {
      // To test (3), generate a new state with the binding added.  If it is
      // the same state, then it escapes (since the store cannot represent
      // the binding).
      // Do this only if we know that the store is not supposed to generate the
      // same state.
      SVal StoredVal = state->getSVal(regionLoc->getRegion());
      if (StoredVal != val)
        escapes = (state == (state->bindLoc(*regionLoc, val)));
    }
    if (!escapes) {
      // Case 4: We do not currently model what happens when a symbol is
      // assigned to a struct field, so be conservative here and let the symbol
      // go. TODO: This could definitely be improved upon.
      escapes = !isa<VarRegion>(regionLoc->getRegion());
    }
  }

  // If our store can represent the binding and we aren't storing to something
  // that doesn't have local storage then just return and have the simulation
  // state continue as is.
  if (!escapes)
      return;

  // Otherwise, find all symbols referenced by 'val' that we are tracking
  // and stop tracking them.
  state = state->scanReachableSymbols<StopTrackingCallback>(val).getState();
  C.addTransition(state);
}

ProgramStateRef RetainCountChecker::evalAssume(ProgramStateRef state,
                                                   SVal Cond,
                                                   bool Assumption) const {

  // FIXME: We may add to the interface of evalAssume the list of symbols
  //  whose assumptions have changed.  For now we just iterate through the
  //  bindings and check if any of the tracked symbols are NULL.  This isn't
  //  too bad since the number of symbols we will track in practice are
  //  probably small and evalAssume is only called at branches and a few
  //  other places.
  RefBindingsTy B = state->get<RefBindings>();

  if (B.isEmpty())
    return state;

  bool changed = false;
  RefBindingsTy::Factory &RefBFactory = state->get_context<RefBindings>();

  for (RefBindingsTy::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    // Check if the symbol is null stop tracking the symbol.
    ConstraintManager &CMgr = state->getConstraintManager();
    ConditionTruthVal AllocFailed = CMgr.isNull(state, I.getKey());
    if (AllocFailed.isConstrainedTrue()) {
      changed = true;
      B = RefBFactory.remove(B, I.getKey());
    }
  }

  if (changed)
    state = state->set<RefBindings>(B);

  return state;
}

ProgramStateRef 
RetainCountChecker::checkRegionChanges(ProgramStateRef state,
                            const StoreManager::InvalidatedSymbols *invalidated,
                                    ArrayRef<const MemRegion *> ExplicitRegions,
                                    ArrayRef<const MemRegion *> Regions,
                                    const CallEvent *Call) const {
  if (!invalidated)
    return state;

  llvm::SmallPtrSet<SymbolRef, 8> WhitelistedSymbols;
  for (ArrayRef<const MemRegion *>::iterator I = ExplicitRegions.begin(),
       E = ExplicitRegions.end(); I != E; ++I) {
    if (const SymbolicRegion *SR = (*I)->StripCasts()->getAs<SymbolicRegion>())
      WhitelistedSymbols.insert(SR->getSymbol());
  }

  for (StoreManager::InvalidatedSymbols::const_iterator I=invalidated->begin(),
       E = invalidated->end(); I!=E; ++I) {
    SymbolRef sym = *I;
    if (WhitelistedSymbols.count(sym))
      continue;
    // Remove any existing reference-count binding.
    state = removeRefBinding(state, sym);
  }
  return state;
}

//===----------------------------------------------------------------------===//
// Handle dead symbols and end-of-path.
//===----------------------------------------------------------------------===//

std::pair<ExplodedNode *, ProgramStateRef >
RetainCountChecker::handleAutoreleaseCounts(ProgramStateRef state, 
                                            ExplodedNode *Pred,
                                            const ProgramPointTag *Tag,
                                            CheckerContext &Ctx,
                                            SymbolRef Sym, RefVal V) const {
  unsigned ACnt = V.getAutoreleaseCount();

  // No autorelease counts?  Nothing to be done.
  if (!ACnt)
    return std::make_pair(Pred, state);

  assert(!Ctx.isObjCGCEnabled() && "Autorelease counts in GC mode?");
  unsigned Cnt = V.getCount();

  // FIXME: Handle sending 'autorelease' to already released object.

  if (V.getKind() == RefVal::ReturnedOwned)
    ++Cnt;

  if (ACnt <= Cnt) {
    if (ACnt == Cnt) {
      V.clearCounts();
      if (V.getKind() == RefVal::ReturnedOwned)
        V = V ^ RefVal::ReturnedNotOwned;
      else
        V = V ^ RefVal::NotOwned;
    } else {
      V.setCount(Cnt - ACnt);
      V.setAutoreleaseCount(0);
    }
    state = setRefBinding(state, Sym, V);
    ExplodedNode *N = Ctx.addTransition(state, Pred, Tag);
    if (N == 0)
      state = 0;
    return std::make_pair(N, state);
  }

  // Woah!  More autorelease counts then retain counts left.
  // Emit hard error.
  V = V ^ RefVal::ErrorOverAutorelease;
  state = setRefBinding(state, Sym, V);

  ExplodedNode *N = Ctx.generateSink(state, Pred, Tag);
  if (N) {
    SmallString<128> sbuf;
    llvm::raw_svector_ostream os(sbuf);
    os << "Object over-autoreleased: object was sent -autorelease ";
    if (V.getAutoreleaseCount() > 1)
      os << V.getAutoreleaseCount() << " times ";
    os << "but the object has a +" << V.getCount() << " retain count";

    if (!overAutorelease)
      overAutorelease.reset(new OverAutorelease());

    const LangOptions &LOpts = Ctx.getASTContext().getLangOpts();
    CFRefReport *report =
      new CFRefReport(*overAutorelease, LOpts, /* GCEnabled = */ false,
                      SummaryLog, N, Sym, os.str());
    Ctx.emitReport(report);
  }

  return std::make_pair((ExplodedNode *)0, (ProgramStateRef )0);
}

ProgramStateRef 
RetainCountChecker::handleSymbolDeath(ProgramStateRef state,
                                      SymbolRef sid, RefVal V,
                                    SmallVectorImpl<SymbolRef> &Leaked) const {
  bool hasLeak = false;
  if (V.isOwned())
    hasLeak = true;
  else if (V.isNotOwned() || V.isReturnedOwned())
    hasLeak = (V.getCount() > 0);

  if (!hasLeak)
    return removeRefBinding(state, sid);

  Leaked.push_back(sid);
  return setRefBinding(state, sid, V ^ RefVal::ErrorLeak);
}

ExplodedNode *
RetainCountChecker::processLeaks(ProgramStateRef state,
                                 SmallVectorImpl<SymbolRef> &Leaked,
                                 CheckerContext &Ctx,
                                 ExplodedNode *Pred) const {
  if (Leaked.empty())
    return Pred;

  // Generate an intermediate node representing the leak point.
  ExplodedNode *N = Ctx.addTransition(state, Pred);

  if (N) {
    for (SmallVectorImpl<SymbolRef>::iterator
         I = Leaked.begin(), E = Leaked.end(); I != E; ++I) {

      const LangOptions &LOpts = Ctx.getASTContext().getLangOpts();
      bool GCEnabled = Ctx.isObjCGCEnabled();
      CFRefBug *BT = Pred ? getLeakWithinFunctionBug(LOpts, GCEnabled)
                          : getLeakAtReturnBug(LOpts, GCEnabled);
      assert(BT && "BugType not initialized.");

      CFRefLeakReport *report = new CFRefLeakReport(*BT, LOpts, GCEnabled, 
                                                    SummaryLog, N, *I, Ctx);
      Ctx.emitReport(report);
    }
  }

  return N;
}

void RetainCountChecker::checkEndPath(CheckerContext &Ctx) const {
  ProgramStateRef state = Ctx.getState();
  RefBindingsTy B = state->get<RefBindings>();
  ExplodedNode *Pred = Ctx.getPredecessor();

  for (RefBindingsTy::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    llvm::tie(Pred, state) = handleAutoreleaseCounts(state, Pred, /*Tag=*/0,
                                                     Ctx, I->first, I->second);
    if (!state)
      return;
  }

  // If the current LocationContext has a parent, don't check for leaks.
  // We will do that later.
  // FIXME: we should instead check for imbalances of the retain/releases,
  // and suggest annotations.
  if (Ctx.getLocationContext()->getParent())
    return;
  
  B = state->get<RefBindings>();
  SmallVector<SymbolRef, 10> Leaked;

  for (RefBindingsTy::iterator I = B.begin(), E = B.end(); I != E; ++I)
    state = handleSymbolDeath(state, I->first, I->second, Leaked);

  processLeaks(state, Leaked, Ctx, Pred);
}

const ProgramPointTag *
RetainCountChecker::getDeadSymbolTag(SymbolRef sym) const {
  const SimpleProgramPointTag *&tag = DeadSymbolTags[sym];
  if (!tag) {
    SmallString<64> buf;
    llvm::raw_svector_ostream out(buf);
    out << "RetainCountChecker : Dead Symbol : ";
    sym->dumpToStream(out);
    tag = new SimpleProgramPointTag(out.str());
  }
  return tag;  
}

void RetainCountChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                          CheckerContext &C) const {
  ExplodedNode *Pred = C.getPredecessor();

  ProgramStateRef state = C.getState();
  RefBindingsTy B = state->get<RefBindings>();

  // Update counts from autorelease pools
  for (SymbolReaper::dead_iterator I = SymReaper.dead_begin(),
       E = SymReaper.dead_end(); I != E; ++I) {
    SymbolRef Sym = *I;
    if (const RefVal *T = B.lookup(Sym)){
      // Use the symbol as the tag.
      // FIXME: This might not be as unique as we would like.
      const ProgramPointTag *Tag = getDeadSymbolTag(Sym);
      llvm::tie(Pred, state) = handleAutoreleaseCounts(state, Pred, Tag, C,
                                                       Sym, *T);
      if (!state)
        return;
    }
  }

  B = state->get<RefBindings>();
  SmallVector<SymbolRef, 10> Leaked;

  for (SymbolReaper::dead_iterator I = SymReaper.dead_begin(),
       E = SymReaper.dead_end(); I != E; ++I) {
    if (const RefVal *T = B.lookup(*I))
      state = handleSymbolDeath(state, *I, *T, Leaked);
  }

  Pred = processLeaks(state, Leaked, C, Pred);

  // Did we cache out?
  if (!Pred)
    return;

  // Now generate a new node that nukes the old bindings.
  RefBindingsTy::Factory &F = state->get_context<RefBindings>();

  for (SymbolReaper::dead_iterator I = SymReaper.dead_begin(),
       E = SymReaper.dead_end(); I != E; ++I)
    B = F.remove(B, *I);

  state = state->set<RefBindings>(B);
  C.addTransition(state, Pred);
}

void RetainCountChecker::printState(raw_ostream &Out, ProgramStateRef State,
                                    const char *NL, const char *Sep) const {

  RefBindingsTy B = State->get<RefBindings>();

  if (!B.isEmpty())
    Out << Sep << NL;

  for (RefBindingsTy::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    Out << I->first << " : ";
    I->second.print(Out);
    Out << NL;
  }
}

//===----------------------------------------------------------------------===//
// Checker registration.
//===----------------------------------------------------------------------===//

void ento::registerRetainCountChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<RetainCountChecker>();
}

