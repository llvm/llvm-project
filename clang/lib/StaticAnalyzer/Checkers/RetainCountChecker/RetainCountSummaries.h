//=== RetainCountSummaries.h - Checks for leaks and other issues -*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines summaries implementation for RetainCountChecker, which
//  implements a reference count checker for Core Foundation and Cocoa
//  on (Mac OS X).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_RETAINCOUNTCHECKER_SUMMARY_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_RETAINCOUNTCHECKER_SUMMARY_H

#include "../ClangSACheckers.h"
#include "../AllocationDiagnostics.h"
#include "../SelectorExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ParentMap.h"
#include "clang/StaticAnalyzer/Checkers/ObjCRetainCount.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/STLExtras.h"

//===----------------------------------------------------------------------===//
// Adapters for FoldingSet.
//===----------------------------------------------------------------------===//

using namespace clang;
using namespace ento;
using namespace objc_retain;

namespace clang {
namespace ento {
namespace retaincountchecker {

/// A key identifying a summary.
class ObjCSummaryKey {
  IdentifierInfo* II;
  Selector S;
public:
  ObjCSummaryKey(IdentifierInfo* ii, Selector s)
    : II(ii), S(s) {}

  ObjCSummaryKey(const ObjCInterfaceDecl *d, Selector s)
    : II(d ? d->getIdentifier() : nullptr), S(s) {}

  ObjCSummaryKey(Selector s)
    : II(nullptr), S(s) {}

  IdentifierInfo *getIdentifier() const { return II; }
  Selector getSelector() const { return S; }
};

} // end namespace retaincountchecker
} // end namespace ento
} // end namespace clang

namespace llvm {
using namespace retaincountchecker;

template <> struct FoldingSetTrait<ArgEffect> {
static inline void Profile(const ArgEffect X, FoldingSetNodeID &ID) {
  ID.AddInteger((unsigned) X);
}
};
template <> struct FoldingSetTrait<RetEffect> {
  static inline void Profile(const RetEffect &X, FoldingSetNodeID &ID) {
    ID.AddInteger((unsigned) X.getKind());
    ID.AddInteger((unsigned) X.getObjKind());
}
};

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

} // end llvm namespace


namespace clang {
namespace ento {
namespace retaincountchecker {

/// ArgEffects summarizes the effects of a function/method call on all of
/// its arguments.
typedef llvm::ImmutableMap<unsigned, ArgEffect> ArgEffects;

/// Summary for a function with respect to ownership changes.
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


  static bool isRetain(const FunctionDecl *FD, StringRef FName) {
    return FName.startswith_lower("retain") || FName.endswith_lower("retain");
  }

  static bool isRelease(const FunctionDecl *FD, StringRef FName) {
    return FName.startswith_lower("release") || FName.endswith_lower("release");
  }

  static bool isAutorelease(const FunctionDecl *FD, StringRef FName) {
    return FName.startswith_lower("autorelease") ||
           FName.endswith_lower("autorelease");
  }

  static bool hasRCAnnotation(const Decl *D, StringRef rcAnnotation) {
    for (const auto *Ann : D->specific_attrs<AnnotateAttr>()) {
      if (Ann->getAnnotation() == rcAnnotation)
        return true;
    }
    return false;
  }

  static bool isTrustedReferenceCountImplementation(const FunctionDecl *FD) {
    return hasRCAnnotation(FD, "rc_ownership_trusted_implementation");
  }

private:
  ArgEffects getArgEffects() const { return Args; }
  ArgEffect getDefaultArgEffect() const { return DefaultArgEffect; }

  friend class RetainSummaryManager;
  friend class RetainCountChecker;
};

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
      return nullptr;

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
        return nullptr;
    }

    // Cache the summary with original key to make the next lookup faster
    // and return the iterator.
    const RetainSummary *Summ = I->second;
    M[K] = Summ;
    return Summ;
  }

  const RetainSummary *find(IdentifierInfo* II, Selector S) {
    // FIXME: Class method lookup.  Right now we don't have a good way
    // of going between IdentifierInfo* and the class hierarchy.
    MapTy::iterator I = M.find(ObjCSummaryKey(II, S));

    if (I == M.end())
      I = M.find(ObjCSummaryKey(S));

    return I == M.end() ? nullptr : I->second;
  }

  const RetainSummary *& operator[](ObjCSummaryKey K) {
    return M[K];
  }

  const RetainSummary *& operator[](Selector S) {
    return M[ ObjCSummaryKey(S) ];
  }
};

class RetainSummaryManager {
  typedef llvm::DenseMap<const FunctionDecl*, const RetainSummary *>
          FuncSummariesTy;

  typedef ObjCSummaryCache ObjCMethodSummariesTy;

  typedef llvm::FoldingSetNodeWrapper<RetainSummary> CachedSummaryNode;

  /// Ctx - The ASTContext object for the analyzed ASTs.
  ASTContext &Ctx;

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

  /// getArgEffects - Returns a persistent ArgEffects object based on the
  ///  data in ScratchArgs.
  ArgEffects getArgEffects();

  enum UnaryFuncKind { cfretain, cfrelease, cfautorelease };

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

  template <typename... Keywords>
  void addMethodSummary(IdentifierInfo *ClsII, ObjCMethodSummariesTy &Summaries,
                        const RetainSummary *Summ, Keywords *... Kws) {
    Selector S = getKeywordSelector(Ctx, Kws...);
    Summaries[ObjCSummaryKey(ClsII, S)] = Summ;
  }

  template <typename... Keywords>
  void addInstMethSummary(const char *Cls, const RetainSummary *Summ,
                          Keywords *... Kws) {
    addMethodSummary(&Ctx.Idents.get(Cls), ObjCMethodSummaries, Summ, Kws...);
  }

  template <typename... Keywords>
  void addClsMethSummary(const char *Cls, const RetainSummary *Summ,
                         Keywords *... Kws) {
    addMethodSummary(&Ctx.Idents.get(Cls), ObjCClassMethodSummaries, Summ,
                     Kws...);
  }

  template <typename... Keywords>
  void addClsMethSummary(IdentifierInfo *II, const RetainSummary *Summ,
                         Keywords *... Kws) {
    addMethodSummary(II, ObjCClassMethodSummaries, Summ, Kws...);
  }

public:
  RetainSummaryManager(ASTContext &ctx, bool usesARC)
   : Ctx(ctx),
     ARCEnabled(usesARC),
     AF(BPAlloc), ScratchArgs(AF.getEmptyMap()),
     ObjCAllocRetE(usesARC ? RetEffect::MakeNotOwned(RetEffect::ObjC)
                               : RetEffect::MakeOwned(RetEffect::ObjC)),
     ObjCInitRetE(usesARC ? RetEffect::MakeNotOwned(RetEffect::ObjC)
                               : RetEffect::MakeOwnedWhenTrackedReceiver()) {
    InitializeClassMethodSummaries();
    InitializeMethodSummaries();
  }

  const RetainSummary *getSummary(const CallEvent &Call,
                                  ProgramStateRef State = nullptr);

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
    QualType ResultTy = MD->getReturnType();

    ObjCMethodSummariesTy *CachedSummaries;
    if (MD->isInstanceMethod())
      CachedSummaries = &ObjCMethodSummaries;
    else
      CachedSummaries = &ObjCClassMethodSummaries;

    return getMethodSummary(S, ID, MD, ResultTy, *CachedSummaries);
  }

  const RetainSummary *getStandardMethodSummary(const ObjCMethodDecl *MD,
                                                Selector S, QualType RetTy);

  /// Determine if there is a special return effect for this function or method.
  Optional<RetEffect> getRetEffectFromAnnotations(QualType RetTy,
                                                  const Decl *D);

  void updateSummaryFromAnnotations(const RetainSummary *&Summ,
                                    const ObjCMethodDecl *MD);

  void updateSummaryFromAnnotations(const RetainSummary *&Summ,
                                    const FunctionDecl *FD);

  void updateSummaryForCall(const RetainSummary *&Summ,
                            const CallEvent &Call);

  bool isARCEnabled() const { return ARCEnabled; }

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

} // end namespace retaincountchecker
} // end namespace ento
} // end namespace clang

#endif
