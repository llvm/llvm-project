//===- Lifetimecheck.cpp - emit diagnostic checks for lifetime violations -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"

#include <functional>

using namespace mlir;
using namespace cir;

namespace {

struct LocOrdering {
  bool operator()(mlir::Location L1, mlir::Location L2) const {
    return std::less<const void *>()(L1.getAsOpaquePointer(),
                                     L2.getAsOpaquePointer());
  }
};

struct LifetimeCheckPass : public LifetimeCheckBase<LifetimeCheckPass> {
  LifetimeCheckPass() = default;
  void runOnOperation() override;

  void checkOperation(Operation *op);
  void checkFunc(cir::FuncOp fnOp);
  void checkBlock(Block &block);

  void checkRegionWithScope(Region &region);
  void checkRegion(Region &region);

  void checkIf(IfOp op);
  void checkSwitch(SwitchOp op);
  void checkLoop(LoopOp op);
  void checkAlloca(AllocaOp op);
  void checkStore(StoreOp op);
  void checkLoad(LoadOp op);
  void checkCall(CallOp callOp);
  void checkAwait(AwaitOp awaitOp);
  void checkReturn(ReturnOp retOp);

  // FIXME: classify tasks and lambdas prior to check ptr deref
  // and pass down an enum.
  void checkPointerDeref(mlir::Value addr, mlir::Location loc,
                         bool forRetLambda = false);
  void checkCoroTaskStore(StoreOp storeOp);
  void checkLambdaCaptureStore(StoreOp storeOp);
  void trackCallToCoroutine(CallOp callOp);

  void checkCtor(CallOp callOp, const clang::CXXConstructorDecl *ctor);
  void checkMoveAssignment(CallOp callOp, const clang::CXXMethodDecl *m);
  void checkCopyAssignment(CallOp callOp, const clang::CXXMethodDecl *m);
  void checkNonConstUseOfOwner(mlir::Value ownerAddr, mlir::Location loc);
  void checkOperators(CallOp callOp, const clang::CXXMethodDecl *m);
  void checkOtherMethodsAndFunctions(CallOp callOp,
                                     const clang::CXXMethodDecl *m);
  void checkForOwnerAndPointerArguments(CallOp callOp, unsigned firstArgIdx);

  // Tracks current module.
  ModuleOp theModule;
  // Track current function under analysis
  std::optional<FuncOp> currFunc;

  // Common helpers.
  bool isCtorInitPointerFromOwner(CallOp callOp,
                                  const clang::CXXConstructorDecl *ctor);
  bool isNonConstUseOfOwner(CallOp callOp, const clang::CXXMethodDecl *m);
  bool isOwnerOrPointerClassMethod(mlir::Value firstParam,
                                   const clang::CXXMethodDecl *m);

  // Diagnostic helpers.
  void emitInvalidHistory(mlir::InFlightDiagnostic &D, mlir::Value histKey,
                          mlir::Location warningLoc, bool forRetLambda);

  ///
  /// Pass options handling
  /// ---------------------

  struct Options {
    enum : unsigned {
      None = 0,
      // Emit pset remarks only detecting invalid derefs
      RemarkPsetInvalid = 1,
      // Emit pset remarks for all derefs
      RemarkPsetAlways = 1 << 1,
      RemarkAll = 1 << 2,
      HistoryNull = 1 << 3,
      HistoryInvalid = 1 << 4,
      HistoryAll = 1 << 5,
    };
    unsigned val = None;
    unsigned histLimit = 1;
    bool isOptionsParsed = false;

    void parseOptions(ArrayRef<StringRef> remarks, ArrayRef<StringRef> hist,
                      unsigned hist_limit) {
      if (isOptionsParsed)
        return;

      for (auto &remark : remarks) {
        val |= StringSwitch<unsigned>(remark)
                   .Case("pset-invalid", RemarkPsetInvalid)
                   .Case("pset-always", RemarkPsetAlways)
                   .Case("all", RemarkAll)
                   .Default(None);
      }
      for (auto &h : hist) {
        val |= StringSwitch<unsigned>(h)
                   .Case("invalid", HistoryInvalid)
                   .Case("null", HistoryNull)
                   .Case("all", HistoryAll)
                   .Default(None);
      }
      histLimit = hist_limit;
      isOptionsParsed = true;
    }

    void parseOptions(LifetimeCheckPass &pass) {
      SmallVector<llvm::StringRef, 4> remarks;
      SmallVector<llvm::StringRef, 4> hists;

      for (auto &r : pass.remarksList)
        remarks.push_back(r);

      for (auto &h : pass.historyList)
        hists.push_back(h);

      parseOptions(remarks, hists, pass.historyLimit);
    }

    bool emitRemarkAll() { return val & RemarkAll; }
    bool emitRemarkPsetInvalid() {
      return emitRemarkAll() || val & RemarkPsetInvalid;
    }
    bool emitRemarkPsetAlways() {
      return emitRemarkAll() || val & RemarkPsetAlways;
    }

    bool emitHistoryAll() { return val & HistoryAll; }
    bool emitHistoryNull() { return emitHistoryAll() || val & HistoryNull; }
    bool emitHistoryInvalid() {
      return emitHistoryAll() || val & HistoryInvalid;
    }
  } opts;

  ///
  /// State
  /// -----

  // Represents the state of an element in a pointer set (pset)
  struct State {
    using DataTy = enum {
      Invalid,
      NullPtr,
      Global,
      // FIXME: currently only supports one level of OwnedBy!
      OwnedBy,
      LocalValue,
      NumKindsMinusOne = LocalValue
    };
    State() { val.setInt(Invalid); }
    State(DataTy d) { val.setInt(d); }
    State(mlir::Value v, DataTy d = LocalValue) {
      assert((d == LocalValue || d == OwnedBy) && "expected value or owned");
      val.setPointerAndInt(v, d);
    }

    static constexpr int KindBits = 3;
    static_assert((1 << KindBits) > NumKindsMinusOne,
                  "Not enough room for kind!");
    llvm::PointerIntPair<mlir::Value, KindBits> val;

    /// Provide less/equal than operator for sorting / set ops.
    bool operator<(const State &RHS) const {
      // FIXME: note that this makes the ordering non-deterministic, do
      // we really care?
      if (hasValue() && RHS.hasValue())
        return val.getPointer().getAsOpaquePointer() <
               RHS.val.getPointer().getAsOpaquePointer();
      return val.getInt() < RHS.val.getInt();
    }
    bool operator==(const State &RHS) const {
      if (hasValue() && RHS.hasValue())
        return val.getPointer() == RHS.val.getPointer();
      return val.getInt() == RHS.val.getInt();
    }

    bool isLocalValue() const { return val.getInt() == LocalValue; }
    bool isOwnedBy() const { return val.getInt() == OwnedBy; }
    bool hasValue() const { return isLocalValue() || isOwnedBy(); }

    mlir::Value getData() const {
      assert(hasValue() && "data type does not hold a mlir::Value");
      return val.getPointer();
    }

    void dump(llvm::raw_ostream &OS = llvm::errs(), int ownedGen = 0);

    static State getInvalid() { return {Invalid}; }
    static State getNullPtr() { return {NullPtr}; }
    static State getLocalValue(mlir::Value v) { return {v, LocalValue}; }
    static State getOwnedBy(mlir::Value v) { return {v, State::OwnedBy}; }
  };

  ///
  /// Invalid and null history tracking
  /// ---------------------------------
  enum InvalidStyle {
    Unknown,
    EndOfScope,
    NotInitialized,
    MovedFrom,
    NonConstUseOfOwner,
  };

  struct InvalidHistEntry {
    InvalidStyle style = Unknown;
    std::optional<mlir::Location> loc;
    std::optional<mlir::Value> val;
    InvalidHistEntry() = default;
    InvalidHistEntry(InvalidStyle s, std::optional<mlir::Location> l,
                     std::optional<mlir::Value> v)
        : style(s), loc(l), val(v) {}
  };

  struct InvalidHist {
    llvm::SmallVector<InvalidHistEntry, 8> entries;
    void add(mlir::Value ptr, InvalidStyle invalidStyle, mlir::Location loc,
             std::optional<mlir::Value> val = {}) {
      entries.emplace_back(InvalidHistEntry(invalidStyle, loc, val));
    }
  };

  llvm::DenseMap<mlir::Value, InvalidHist> invalidHist;

  using PMapNullHistType =
      llvm::DenseMap<mlir::Value, std::optional<mlir::Location>>;
  PMapNullHistType pmapNullHist;

  ///
  /// Pointer Map and Pointer Set
  /// ---------------------------

  using PSetType = llvm::SmallSet<State, 4>;
  // FIXME: this should be a ScopedHashTable for consistency.
  using PMapType = llvm::DenseMap<mlir::Value, PSetType>;

  // FIXME: we probably don't need to track it at this level, perhaps
  // just tracking at the scope level should be enough?
  PMapType *currPmap = nullptr;
  PMapType &getPmap() { return *currPmap; }
  void markPsetInvalid(mlir::Value ptr, InvalidStyle invalidStyle,
                       mlir::Location loc,
                       std::optional<mlir::Value> extraVal = {}) {
    auto &pset = getPmap()[ptr];

    // If pset is already invalid, don't bother.
    if (pset.count(State::getInvalid()))
      return;

    // 2.3 - putting invalid into pset(x) is said to invalidate it
    pset.insert(State::getInvalid());
    invalidHist[ptr].add(ptr, invalidStyle, loc, extraVal);
  }

  void joinPmaps(SmallVectorImpl<PMapType> &pmaps);

  // Provides p1179's 'KILL' functionality. See implementation for more
  // information.
  void kill(const State &s, InvalidStyle invalidStyle, mlir::Location loc);
  void killInPset(mlir::Value ptrKey, const State &s, InvalidStyle invalidStyle,
                  mlir::Location loc, std::optional<mlir::Value> extraVal);

  // Local pointers
  SmallPtrSet<mlir::Value, 8> ptrs;

  // Local owners. We use a map instead of a set to track the current generation
  // for this owner type internal pointee's. For instance, this allows tracking
  // subsequent reuse of owner storage when a non-const use happens.
  DenseMap<mlir::Value, unsigned> owners;
  void addOwner(mlir::Value o) {
    assert(!owners.count(o) && "already tracked");
    owners[o] = 0;
  }
  void incOwner(mlir::Value o) {
    assert(owners.count(o) && "entry expected");
    owners[o]++;
  }

  // Aggregates and exploded fields.
  using ExplodedFieldsTy = llvm::SmallSet<unsigned, 4>;
  DenseMap<mlir::Value, ExplodedFieldsTy> aggregates;
  void addAggregate(mlir::Value a, SmallVectorImpl<unsigned> &fields) {
    assert(!aggregates.count(a) && "already tracked");
    aggregates[a].insert(fields.begin(), fields.end());
  }

  // Useful helpers for debugging
  void printPset(PSetType &pset, llvm::raw_ostream &OS = llvm::errs());
  LLVM_DUMP_METHOD void dumpPmap(PMapType &pmap);
  LLVM_DUMP_METHOD void dumpCurrentPmap();

  ///
  /// Coroutine tasks (promise_type)
  /// ------------------------------

  // Track types we already know to be a coroutine task (promise_type)
  llvm::DenseMap<mlir::Type, bool> IsTaskTyCache;
  // Is the type associated with taskVal a coroutine task? Uses IsTaskTyCache
  // or compute it from associated AST node.
  bool isTaskType(mlir::Value taskVal);
  // Addresses of coroutine Tasks found in the current function.
  SmallPtrSet<mlir::Value, 8> tasks;
  // Since coawait encapsulates several calls to a promise, do not emit
  // the same warning multiple times, e.g. under the same coawait.
  llvm::SmallSet<mlir::Location, 8, LocOrdering> emittedDanglingTasks;

  ///
  /// Lambdas
  /// -------

  // Track types we already know to be a lambda
  llvm::DenseMap<mlir::Type, bool> IsLambdaTyCache;
  // Check if a given cir type is a struct containing a lambda
  bool isLambdaType(mlir::Type ty);
  // Get the lambda struct from a member access to it.
  mlir::Value getLambdaFromMemberAccess(mlir::Value addr);

  ///
  /// Scope, context and guards
  /// -------------------------

  // Represents the scope context for IR operations (cir.scope, cir.if,
  // then/else regions, etc). Tracks the declaration of variables in the current
  // local scope.
  struct LexicalScopeContext {
    unsigned Depth = 0;
    LexicalScopeContext() = delete;

    llvm::PointerUnion<mlir::Region *, mlir::Operation *> parent;
    LexicalScopeContext(mlir::Region *R) : parent(R) {}
    LexicalScopeContext(mlir::Operation *Op) : parent(Op) {}
    ~LexicalScopeContext() = default;

    // Track all local values added in this scope
    SmallPtrSet<mlir::Value, 4> localValues;

    // Track the result of temporaries with coroutine call results,
    // they are used to initialize a task.
    //
    // Value must come directly out of a cir.call to a cir.func which
    // is a coroutine.
    SmallPtrSet<mlir::Value, 2> localTempTasks;

    // Track seen lambdas that escape out of the current scope
    // (e.g. lambdas returned out of functions).
    DenseMap<mlir::Value, mlir::Location> localRetLambdas;

    LLVM_DUMP_METHOD void dumpLocalValues();
  };

  class LexicalScopeGuard {
    LifetimeCheckPass &Pass;
    LexicalScopeContext *OldVal = nullptr;

  public:
    LexicalScopeGuard(LifetimeCheckPass &p, LexicalScopeContext *L) : Pass(p) {
      if (Pass.currScope) {
        OldVal = Pass.currScope;
        L->Depth++;
      }
      Pass.currScope = L;
    }

    LexicalScopeGuard(const LexicalScopeGuard &) = delete;
    LexicalScopeGuard &operator=(const LexicalScopeGuard &) = delete;
    LexicalScopeGuard &operator=(LexicalScopeGuard &&other) = delete;

    void cleanup();
    void restore() { Pass.currScope = OldVal; }
    ~LexicalScopeGuard() {
      cleanup();
      restore();
    }
  };

  class PmapGuard {
    LifetimeCheckPass &Pass;
    PMapType *OldVal = nullptr;

  public:
    PmapGuard(LifetimeCheckPass &lcp, PMapType *L) : Pass(lcp) {
      if (Pass.currPmap) {
        OldVal = Pass.currPmap;
      }
      Pass.currPmap = L;
    }

    PmapGuard(const PmapGuard &) = delete;
    PmapGuard &operator=(const PmapGuard &) = delete;
    PmapGuard &operator=(PmapGuard &&other) = delete;

    void restore() { Pass.currPmap = OldVal; }
    ~PmapGuard() { restore(); }
  };

  LexicalScopeContext *currScope = nullptr;

  ///
  /// AST related
  /// -----------

  std::optional<clang::ASTContext *> astCtx;
  void setASTContext(clang::ASTContext *c) { astCtx = c; }
};
} // namespace

static StringRef getVarNameFromValue(mlir::Value v) {
  if (auto allocaOp = dyn_cast<AllocaOp>(v.getDefiningOp()))
    return allocaOp.getName();
  assert(0 && "how did it get here?");
  return "";
}

static Location getEndLoc(Location loc, int idx = 1) {
  auto fusedLoc = loc.dyn_cast<FusedLoc>();
  if (!fusedLoc)
    return loc;
  return fusedLoc.getLocations()[idx];
}

static Location getEndLocForHist(Operation *Op) {
  return getEndLoc(Op->getLoc());
}

static Location getEndLocIf(IfOp ifOp, Region *R) {
  assert(ifOp && "what other regions create their own scope?");
  if (&ifOp.getThenRegion() == R)
    return getEndLoc(ifOp.getLoc());
  return getEndLoc(ifOp.getLoc(), /*idx=*/3);
}

static Location getEndLocForHist(Region *R) {
  auto parentOp = R->getParentOp();
  if (isa<IfOp>(parentOp))
    return getEndLocIf(cast<IfOp>(parentOp), R);
  if (isa<FuncOp>(parentOp))
    return getEndLoc(parentOp->getLoc());
  llvm_unreachable("what other regions create their own scope?");
}

static Location getEndLocForHist(LifetimeCheckPass::LexicalScopeContext &lsc) {
  assert(!lsc.parent.isNull() && "shouldn't be null");
  if (lsc.parent.is<Region *>())
    return getEndLocForHist(lsc.parent.get<Region *>());
  assert(lsc.parent.is<Operation *>() &&
         "Only support operation beyond this point");
  return getEndLocForHist(lsc.parent.get<Operation *>());
}

void LifetimeCheckPass::killInPset(mlir::Value ptrKey, const State &s,
                                   InvalidStyle invalidStyle,
                                   mlir::Location loc,
                                   std::optional<mlir::Value> extraVal) {
  auto &pset = getPmap()[ptrKey];
  if (pset.contains(s)) {
    pset.erase(s);
    markPsetInvalid(ptrKey, invalidStyle, loc, extraVal);
  }
}

// 2.3 - KILL(x) means to replace all occurrences of x and x' and x'' (etc.)
// in the pmap with invalid. For example, if pmap is {(p1,{a}), (p2,{a'})},
// KILL(a') would invalidate only p2, and KILL(a) would invalidate both p1 and
// p2.
void LifetimeCheckPass::kill(const State &s, InvalidStyle invalidStyle,
                             mlir::Location loc) {
  assert(s.hasValue() && "does not know how to kill other data types");
  mlir::Value v = s.getData();
  std::optional<mlir::Value> extraVal;
  if (invalidStyle == InvalidStyle::EndOfScope)
    extraVal = v;

  for (auto &mapEntry : getPmap()) {
    auto ptr = mapEntry.first;

    // We are deleting this entry anyways, nothing to do here.
    if (v == ptr)
      continue;

    // ... replace all occurrences of x and x' and x''. Start with the primes
    // so we first remove uses and then users.
    //
    // FIXME: add x'', x''', etc...
    if (s.isLocalValue() && owners.count(v))
      killInPset(ptr, State::getOwnedBy(v), invalidStyle, loc, extraVal);
    killInPset(ptr, s, invalidStyle, loc, extraVal);
  }

  // Delete the local value from pmap, since its scope has ended.
  if (invalidStyle == InvalidStyle::EndOfScope) {
    owners.erase(v);
    ptrs.erase(v);
    tasks.erase(v);
  }
}

void LifetimeCheckPass::LexicalScopeGuard::cleanup() {
  auto *localScope = Pass.currScope;
  for (auto pointee : localScope->localValues)
    Pass.kill(State::getLocalValue(pointee), InvalidStyle::EndOfScope,
              getEndLocForHist(*localScope));

  // Catch interesting dangling references out of returns.
  for (auto l : localScope->localRetLambdas)
    Pass.checkPointerDeref(l.first, l.second,
                           /*forRetLambda=*/true);
}

void LifetimeCheckPass::checkBlock(Block &block) {
  // Block main role is to hold a list of Operations.
  for (Operation &op : block.getOperations())
    checkOperation(&op);
}

void LifetimeCheckPass::checkRegion(Region &region) {
  for (Block &block : region.getBlocks())
    checkBlock(block);
}

void LifetimeCheckPass::checkRegionWithScope(Region &region) {
  // Add a new scope. Note that as part of the scope cleanup process
  // we apply section 2.3 KILL(x) functionality, turning relevant
  // references invalid.
  LexicalScopeContext lexScope{&region};
  LexicalScopeGuard scopeGuard{*this, &lexScope};
  for (Block &block : region.getBlocks())
    checkBlock(block);
}

void LifetimeCheckPass::checkFunc(cir::FuncOp fnOp) {
  currFunc = fnOp;
  // FIXME: perhaps this should be a function pass, but for now make
  // sure we reset the state before looking at other functions.
  if (currPmap)
    getPmap().clear();
  pmapNullHist.clear();
  invalidHist.clear();

  // Create a new pmap for this function.
  PMapType localPmap{};
  PmapGuard pmapGuard{*this, &localPmap};

  // Add a new scope. Note that as part of the scope cleanup process
  // we apply section 2.3 KILL(x) functionality, turning relevant
  // references invalid.
  for (Region &region : fnOp->getRegions())
    checkRegionWithScope(region);

  // FIXME: store the pmap result for this function, we
  // could do some interesting IPA stuff using this info.
  currFunc.reset();
}

// The join operation between pmap as described in section 2.3.
//
//  JOIN({pmap1,...,pmapN}) =>
//  { (p, pset1(p) U ... U psetN(p) | (p,*) U pmap1 U ... U pmapN }.
//
void LifetimeCheckPass::joinPmaps(SmallVectorImpl<PMapType> &pmaps) {
  for (auto &mapEntry : getPmap()) {
    auto &val = mapEntry.first;

    PSetType joinPset;
    for (auto &pmapOp : pmaps)
      llvm::set_union(joinPset, pmapOp[val]);

    getPmap()[val] = joinPset;
  }
}

void LifetimeCheckPass::checkLoop(LoopOp loopOp) {
  // 2.4.9. Loops
  //
  // A loop is treated as if it were the first two loop iterations unrolled
  // using an if. For example:
  //
  //  for (/*init*/; /*cond*/; /*incr*/)
  //   { /*body*/ }
  //
  // is treated as:
  //
  //  if (/*init*/; /*cond*/)
  //   { /*body*/; /*incr*/ }
  //  if (/*cond*/)
  //   { /*body*/ }
  //
  // See checkIf for additional explanations.
  SmallVector<PMapType, 4> pmapOps;
  SmallVector<Region *, 4> regionsToCheck;

  auto setupLoopRegionsToCheck = [&](bool isSubsequentTaken = false) {
    regionsToCheck.clear();
    switch (loopOp.getKind()) {
    case LoopOpKind::For: {
      regionsToCheck.push_back(&loopOp.getCond());
      regionsToCheck.push_back(&loopOp.getBody());
      if (!isSubsequentTaken)
        regionsToCheck.push_back(&loopOp.getStep());
      break;
    }
    case LoopOpKind::While: {
      regionsToCheck.push_back(&loopOp.getCond());
      regionsToCheck.push_back(&loopOp.getBody());
      break;
    }
    case LoopOpKind::DoWhile: {
      // Note this is the reverse order from While above.
      regionsToCheck.push_back(&loopOp.getBody());
      regionsToCheck.push_back(&loopOp.getCond());
      break;
    }
    }
  };

  // From 2.4.9 "Note":
  //
  // There are only three paths to analyze:
  // (1) never taken (the loop body was not entered)
  pmapOps.push_back(getPmap());

  // (2) first taken (the first pass through the loop body, which begins
  // with the loop entry pmap)
  PMapType loopExitPmap;
  {
    // Intentional copy from loop entry map
    loopExitPmap = getPmap();
    PmapGuard pmapGuard{*this, &loopExitPmap};
    setupLoopRegionsToCheck();
    for (auto *r : regionsToCheck)
      checkRegion(*r);
    pmapOps.push_back(loopExitPmap);
  }

  // (3) and subsequent taken (second or later iteration, which begins with the
  // loop body exit pmap and so takes into account any invalidations performed
  // in the loop body on any path that could affect the next loop).
  //
  // This ensures that a subsequent loop iteration does not use a Pointer that
  // was invalidated during a previous loop iteration.
  //
  // Because this analysis gives the same answer for each block of code (always
  // converges), all loop iterations after the first get the same answer and
  // so we only need to consider the second iteration, and so the analysis
  // algorithm remains linear, single-pass. As an optimization, if the loop
  // entry pmap is the same as the first loop body exit pmap, there is no need
  // to perform the analysis on the second loop iteration; the answer will be
  // the same.
  if (getPmap() != loopExitPmap) {
    // Intentional copy from first taken loop exit pmap
    PMapType otherTakenPmap = loopExitPmap;
    PmapGuard pmapGuard{*this, &otherTakenPmap};
    setupLoopRegionsToCheck(/*isSubsequentTaken=*/true);
    for (auto *r : regionsToCheck)
      checkRegion(*r);
    pmapOps.push_back(otherTakenPmap);
  }

  joinPmaps(pmapOps);
}

void LifetimeCheckPass::checkAwait(AwaitOp awaitOp) {
  // Pretty conservative: assume all regions execute
  // sequencially.
  //
  // FIXME: use branch interface here and only tackle
  // the necessary regions.
  SmallVector<PMapType, 4> pmapOps;

  for (auto r : awaitOp.getRegions()) {
    PMapType regionPmap = getPmap();
    PmapGuard pmapGuard{*this, &regionPmap};
    checkRegion(*r);
    pmapOps.push_back(regionPmap);
  }

  joinPmaps(pmapOps);
}

void LifetimeCheckPass::checkReturn(ReturnOp retOp) {
  // Upon return invalidate all local values. Since some return
  // values might depend on other local address, check for the
  // dangling aspects for this.
  if (retOp.getNumOperands() == 0)
    return;

  auto retTy = retOp.getOperand(0).getType();
  // FIXME: this can be extended to cover more leaking/dandling
  // semantics out of functions.
  if (!isLambdaType(retTy))
    return;

  // The return value is loaded from the return slot before
  // returning.
  auto loadOp = dyn_cast<LoadOp>(retOp.getOperand(0).getDefiningOp());
  assert(loadOp && "expected cir.load");
  if (!isa<AllocaOp>(loadOp.getAddr().getDefiningOp()))
    return;

  // Keep track of interesting lambda.
  assert(!currScope->localRetLambdas.count(loadOp.getAddr()) &&
         "lambda already returned?");
  currScope->localRetLambdas.insert(
      std::make_pair(loadOp.getAddr(), loadOp.getLoc()));
}

void LifetimeCheckPass::checkSwitch(SwitchOp switchOp) {
  // 2.4.7. A switch(cond) is treated as if it were an equivalent series of
  // non-nested if statements with single evaluation of cond; for example:
  //
  //    switch (a) {
  //      case 1:/*1*/
  //      case 2:/*2*/ break;
  //      default:/*3*/
  //    }
  //
  // is treated as:
  //
  //    if (auto& a=a; a==1) {/*1*/}
  //    else if (a==1 || a==2) {/*2*/}
  //    else {/*3*/}.
  //
  // See checkIf for additional explanations.
  SmallVector<PMapType, 2> pmapOps;

  // If there are no regions, pmap is the same.
  if (switchOp.getRegions().empty())
    return;

  auto isCaseFallthroughTerminated = [&](Region &r) {
    assert(r.getBlocks().size() == 1 && "cannot yet handle branches");
    Block &block = r.back();
    assert(!block.empty() && "case regions cannot be empty");

    // FIXME: do something special about return terminated?
    YieldOp y = dyn_cast<YieldOp>(block.back());
    if (!y)
      return false;
    if (y.isFallthrough())
      return true;
    return false;
  };

  auto regions = switchOp.getRegions();
  for (unsigned regionCurrent = 0, regionPastEnd = regions.size();
       regionCurrent != regionPastEnd; ++regionCurrent) {
    // Intentional pmap copy, basis to start new path.
    PMapType locaCasePmap = getPmap();
    PmapGuard pmapGuard{*this, &locaCasePmap};

    // At any given point, fallbacks (if not empty) will increase the
    // number of control-flow possibilities. For each region ending up
    // with a fallback, keep computing the pmap until we hit a region
    // that has a non-fallback terminator for the region.
    unsigned idx = regionCurrent;
    while (idx < regionPastEnd) {
      // Note that for 'if' regions we use checkRegionWithScope, since
      // there are lexical scopes associated with each region, this is
      // not the case for switch's.
      checkRegion(regions[idx]);
      if (!isCaseFallthroughTerminated(regions[idx]))
        break;
      idx++;
    }
    pmapOps.push_back(locaCasePmap);
  }

  joinPmaps(pmapOps);
}

void LifetimeCheckPass::checkIf(IfOp ifOp) {
  // Both then and else create their own lexical scopes, take that into account
  // while checking then/else.
  //
  // This is also the moment where pmaps are joined because flow forks:
  //    pmap(ifOp) = JOIN( pmap(then), pmap(else) )
  //
  // To that intent the pmap is copied out before checking each region and
  // pmap(ifOp) computed after analysing both paths.
  SmallVector<PMapType, 2> pmapOps;

  {
    PMapType localThenPmap = getPmap();
    PmapGuard pmapGuard{*this, &localThenPmap};
    checkRegionWithScope(ifOp.getThenRegion());
    pmapOps.push_back(localThenPmap);
  }

  // In case there's no 'else' branch, the 'else' pmap is the same as
  // prior to the if condition.
  if (!ifOp.getElseRegion().empty()) {
    PMapType localElsePmap = getPmap();
    PmapGuard pmapGuard{*this, &localElsePmap};
    checkRegionWithScope(ifOp.getElseRegion());
    pmapOps.push_back(localElsePmap);
  } else {
    pmapOps.push_back(getPmap());
  }

  joinPmaps(pmapOps);
}

template <class T> bool isStructAndHasAttr(mlir::Type ty) {
  if (!ty.isa<mlir::cir::StructType>())
    return false;
  auto sTy = ty.cast<mlir::cir::StructType>();
  const auto *recordDecl = sTy.getAst()->getAstDecl();
  if (recordDecl->hasAttr<T>())
    return true;
  return false;
}

static bool isOwnerType(mlir::Type ty) {
  // From 2.1:
  //
  // An Owner uniquely owns another object (cannot dangle). An Owner type is
  // expressed using the annotation [[gsl::Owner(DerefType)]] where DerefType is
  // the owned type (and (DerefType) may be omitted and deduced as below). For
  // example:
  //
  // template<class T> class [[gsl::Owner(T)]] my_unique_smart_pointer;
  //
  // TODO: The following standard or other types are treated as-if annotated as
  // Owners, if not otherwise annotated and if not SharedOwners:
  //
  // - Every type that satisfies the standard Container requirements and has a
  // user-provided destructor. (Example: vector.) DerefType is ::value_type.
  // - Every type that provides unary * and has a user-provided destructor.
  // (Example: unique_ptr.) DerefType is the ref-unqualified return type of
  // operator*.
  // - Every type that has a data member or public base class of an Owner type.
  // Additionally, for convenient adoption without modifying existing standard
  // library headers, the following well known standard types are treated as-if
  // annotated as Owners: stack, queue, priority_queue, optional, variant, any,
  // and regex.
  return isStructAndHasAttr<clang::OwnerAttr>(ty);
}

static bool containsPointerElts(mlir::cir::StructType s) {
  auto members = s.getMembers();
  return std::any_of(members.begin(), members.end(), [](mlir::Type t) {
    return t.isa<mlir::cir::PointerType>();
  });
}

static bool isAggregateType(AllocaOp allocaOp) {
  auto t = allocaOp.getAllocaType().dyn_cast<mlir::cir::StructType>();
  if (!t)
    return false;
  // FIXME: For now we handle this in a more naive way: any pointer
  // element we find is enough to consider this an aggregate. But in
  // reality it should be as defined in 2.1:
  //
  // An Aggregate is a type that is not an Indirection and is a class type with
  // public data members none of which are references (& or &&) and no
  // user-provided copy or move operations, and no base class that is not also
  // an Aggregate. The elements of an Aggregate are its public data members.
  return containsPointerElts(t);
}

static bool isPointerType(AllocaOp allocaOp) {
  // From 2.1:
  //
  // A Pointer is not an Owner and provides indirect access to an object it does
  // not own (can dangle). A Pointer type is expressed using the annotation
  // [[gsl::Pointer(DerefType)]] where DerefType is the pointed-to type (and
  // (Dereftype) may be omitted and deduced as below). For example:
  //
  // template<class T> class [[gsl::Pointer(T)]] my_span;
  //
  // TODO: The following standard or other types are treated as-if annotated as
  // Pointer, if not otherwise annotated and if not Owners:
  //
  // - Every type that satisfies the standard Iterator requirements. (Example:
  // regex_iterator.) DerefType is the ref-unqualified return type of operator*.
  // - Every type that satisfies the Ranges TS Range concept. (Example:
  // basic_string_view.) DerefType is the ref-unqualified type of *begin().
  // - Every type that satisfies the following concept. DerefType is the
  // ref-unqualified return type of operator*.
  //
  //  template<typename T> concept
  //  TriviallyCopyableAndNonOwningAndDereferenceable =
  //  std::is_trivially_copyable_v<T> && std::is_copy_constructible_v<T> &&
  //  std::is_copy_assignable_v<T> && requires(T t) { *t; };
  //
  // - Every closure type of a lambda that captures by reference or captures a
  // Pointer by value. DerefType is void.
  // - Every type that has a data member or public base class of a Pointer type.
  // Additionally, for convenient adoption without modifying existing standard
  // library headers, the following well- known standard types are treated as-if
  // annotated as Pointers, in addition to raw pointers and references: ref-
  // erence_wrapper, and vector<bool>::reference.
  if (allocaOp.isPointerType())
    return true;
  return isStructAndHasAttr<clang::PointerAttr>(allocaOp.getAllocaType());
}

void LifetimeCheckPass::checkAlloca(AllocaOp allocaOp) {
  auto addr = allocaOp.getAddr();
  assert(!getPmap().count(addr) && "only one alloca for any given address");
  getPmap()[addr] = {};

  enum TypeCategory {
    Unknown = 0,
    SharedOwner = 1,
    Owner = 1 << 2,
    Pointer = 1 << 3,
    Indirection = 1 << 4,
    Aggregate = 1 << 5,
    Value = 1 << 6,
  };

  auto localStyle = [&]() {
    if (isPointerType(allocaOp))
      return TypeCategory::Pointer;
    if (isOwnerType(allocaOp.getAllocaType()))
      return TypeCategory::Owner;
    if (isAggregateType(allocaOp))
      return TypeCategory::Aggregate;
    return TypeCategory::Value;
  }();

  switch (localStyle) {
  case TypeCategory::Pointer:
    // 2.4.2 - When a non-parameter non-member Pointer p is declared, add
    // (p, {invalid}) to pmap.
    ptrs.insert(addr);
    markPsetInvalid(addr, InvalidStyle::NotInitialized, allocaOp.getLoc());
    break;
  case TypeCategory::Owner:
    // 2.4.2 - When a local Owner x is declared, add (x, {x__1'}) to pmap.
    addOwner(addr);
    getPmap()[addr].insert(State::getOwnedBy(addr));
    currScope->localValues.insert(addr);
    break;
  case TypeCategory::Aggregate: {
    // 2.1 - Aggregates are types we will “explode” (consider memberwise) at
    // local scopes, because the function can operate on the members directly.

    // Explode all pointer members.
    SmallVector<unsigned, 4> fields;
    auto members =
        allocaOp.getAllocaType().cast<mlir::cir::StructType>().getMembers();

    unsigned fieldIdx = 0;
    std::for_each(members.begin(), members.end(), [&](mlir::Type t) {
      auto ptrType = t.dyn_cast<mlir::cir::PointerType>();
      if (ptrType)
        fields.push_back(fieldIdx);
      fieldIdx++;
    });
    addAggregate(addr, fields);

    // Differently from `TypeCategory::Pointer`, initialization for exploded
    // pointer is done lazily, triggered whenever the relevant
    // `cir.struct_element_addr` are seen. This also serves optimization
    // purposes: only track fields that are actually seen.
    break;
  }
  case TypeCategory::Value: {
    // 2.4.2 - When a local Value x is declared, add (x, {x}) to pmap.
    getPmap()[addr].insert(State::getLocalValue(addr));
    currScope->localValues.insert(addr);
    break;
  }
  default:
    llvm_unreachable("NYI");
  }
}

void LifetimeCheckPass::checkCoroTaskStore(StoreOp storeOp) {
  // Given:
  //  auto task = [init task];
  // Extend pset(task) such that:
  //  pset(task) = pset(task) U {any local values used to init task}
  auto taskTmp = storeOp.getValue();
  // FIXME: check it's initialization 'init' attr.
  auto taskAddr = storeOp.getAddr();

  // Take the following coroutine creation pattern:
  //
  //   %task = cir.alloca ...
  //   cir.scope {
  //     %arg0 = cir.alloca ...
  //     ...
  //     %tmp_task = cir.call @corotine_call(%arg0, %arg1, ...)
  //     cir.store %tmp_task, %task
  //     ...
  //   }
  //
  // Bind values that are coming from alloca's (like %arg0 above) to the
  // pset of %task - this effectively leads to some invalidation of %task
  // when %arg0 finishes its lifetime at the end of the enclosing cir.scope.
  if (auto call = dyn_cast<mlir::cir::CallOp>(taskTmp.getDefiningOp())) {
    bool potentialTaintedTask = false;
    for (auto arg : call.getArgOperands()) {
      auto alloca = dyn_cast<mlir::cir::AllocaOp>(arg.getDefiningOp());
      if (alloca && currScope->localValues.count(alloca)) {
        getPmap()[taskAddr].insert(State::getLocalValue(alloca));
        potentialTaintedTask = true;
      }
    }

    // Task are only interesting when there are local addresses leaking
    // via the coroutine creation, only track those.
    if (potentialTaintedTask)
      tasks.insert(taskAddr);
    return;
  }
  llvm_unreachable("expecting cir.call defining op");
}

mlir::Value LifetimeCheckPass::getLambdaFromMemberAccess(mlir::Value addr) {
  auto op = addr.getDefiningOp();
  // FIXME: we likely want to consider more indirections here...
  if (!isa<mlir::cir::StructElementAddr>(op))
    return nullptr;
  auto allocaOp =
      dyn_cast<mlir::cir::AllocaOp>(op->getOperand(0).getDefiningOp());
  if (!allocaOp || !isLambdaType(allocaOp.getAllocaType()))
    return nullptr;
  return allocaOp;
}

void LifetimeCheckPass::checkLambdaCaptureStore(StoreOp storeOp) {
  auto localByRefAddr = storeOp.getValue();
  auto lambdaCaptureAddr = storeOp.getAddr();

  if (!isa_and_nonnull<mlir::cir::AllocaOp>(localByRefAddr.getDefiningOp()))
    return;
  auto lambdaAddr = getLambdaFromMemberAccess(lambdaCaptureAddr);
  if (!lambdaAddr)
    return;

  if (currScope->localValues.count(localByRefAddr))
    getPmap()[lambdaAddr].insert(State::getLocalValue(localByRefAddr));
}

void LifetimeCheckPass::checkStore(StoreOp storeOp) {
  auto addr = storeOp.getAddr();

  // The bulk of the check is done on top of store to pointer categories,
  // which usually represent the most common case.
  //
  // We handle some special local values, like coroutine tasks and lambdas,
  // which could be holding references to things with dangling lifetime.
  if (!ptrs.count(addr)) {
    if (currScope->localTempTasks.count(storeOp.getValue()))
      checkCoroTaskStore(storeOp);
    else
      checkLambdaCaptureStore(storeOp);
    return;
  }

  // Only handle ptrs from here on.

  auto getArrayFromSubscript = [&](PtrStrideOp strideOp) -> mlir::Value {
    auto castOp = dyn_cast<CastOp>(strideOp.getBase().getDefiningOp());
    if (!castOp)
      return {};
    if (castOp.getKind() != cir::CastKind::array_to_ptrdecay)
      return {};
    return castOp.getSrc();
  };

  auto data = storeOp.getValue();
  auto *defOp = data.getDefiningOp();

  // Do not handle block arguments just yet.
  if (!defOp)
    return;

  // 2.4.2 - If the declaration includes an initialization, the
  // initialization is treated as a separate operation
  if (auto cstOp = dyn_cast<ConstantOp>(defOp)) {
    assert(cstOp.isNullPtr() && "not implemented");
    assert(getPmap().count(addr) && "address should always be valid");
    // 2.4.2 - If the initialization is default initialization or zero
    // initialization, set pset(p) = {null}; for example:
    //
    //  int* p; => pset(p) == {invalid}
    //  int* p{}; or string_view p; => pset(p) == {null}.
    //  int *p = nullptr; => pset(p) == {nullptr} => pset(p) == {null}
    getPmap()[addr].clear();
    getPmap()[addr].insert(State::getNullPtr());
    pmapNullHist[addr] = storeOp.getValue().getLoc();
    return;
  }

  if (auto allocaOp = dyn_cast<AllocaOp>(defOp)) {
    // p = &x;
    getPmap()[addr].clear();
    getPmap()[addr].insert(State::getLocalValue(data));
    return;
  }

  if (auto ptrStrideOp = dyn_cast<PtrStrideOp>(defOp)) {
    // p = &a[0];
    auto array = getArrayFromSubscript(ptrStrideOp);
    if (array) {
      getPmap()[addr].clear();
      getPmap()[addr].insert(State::getLocalValue(array));
    }
    return;
  }

  // From here on, some uninterestring store (for now?)
}

void LifetimeCheckPass::checkLoad(LoadOp loadOp) {
  auto addr = loadOp.getAddr();
  // Only interested in checking deference on top of pointer types.
  // Note that usually the use of the invalid address happens at the
  // load or store using the result of this loadOp.
  if (!getPmap().count(addr) || !ptrs.count(addr))
    return;

  if (!loadOp.getIsDeref())
    return;

  checkPointerDeref(addr, loadOp.getLoc());
}

void LifetimeCheckPass::emitInvalidHistory(mlir::InFlightDiagnostic &D,
                                           mlir::Value histKey,
                                           mlir::Location warningLoc,
                                           bool forRetLambda) {
  assert(invalidHist.count(histKey) && "expected invalid hist");
  auto &hist = invalidHist[histKey];
  unsigned limit = opts.histLimit;

  for (int lastIdx = hist.entries.size() - 1; limit > 0 && lastIdx >= 0;
       lastIdx--, limit--) {
    auto &info = hist.entries[lastIdx];

    switch (info.style) {
    case InvalidStyle::NotInitialized: {
      D.attachNote(info.loc) << "uninitialized here";
      break;
    }
    case InvalidStyle::EndOfScope: {
      if (tasks.count(histKey)) {
        StringRef resource = "resource";
        if (auto allocaOp = dyn_cast<AllocaOp>(info.val->getDefiningOp())) {
          if (isLambdaType(allocaOp.getAllocaType()))
            resource = "lambda";
        }
        D.attachNote((*info.val).getLoc())
            << "coroutine bound to " << resource << " with expired lifetime";
        D.attachNote(info.loc) << "at the end of scope or full-expression";
        emittedDanglingTasks.insert(warningLoc);
      } else if (forRetLambda) {
        assert(currFunc && "expected function");
        StringRef parent = currFunc->getLambda() ? "lambda" : "function";
        D.attachNote(info.val->getLoc())
            << "declared here but invalid after enclosing " << parent
            << " ends";
      } else {
        StringRef outOfScopeVarName = getVarNameFromValue(*info.val);
        D.attachNote(info.loc) << "pointee '" << outOfScopeVarName
                               << "' invalidated at end of scope";
      }
      break;
    }
    case InvalidStyle::NonConstUseOfOwner: {
      D.attachNote(info.loc) << "invalidated by non-const use of owner type";
      break;
    }
    default:
      llvm_unreachable("unknown history style");
    }
  }
}

void LifetimeCheckPass::checkPointerDeref(mlir::Value addr, mlir::Location loc,
                                          bool forRetLambda) {
  bool hasInvalid = getPmap()[addr].count(State::getInvalid());
  bool hasNullptr = getPmap()[addr].count(State::getNullPtr());

  auto emitPsetRemark = [&] {
    llvm::SmallString<128> psetStr;
    llvm::raw_svector_ostream Out(psetStr);
    printPset(getPmap()[addr], Out);
    emitRemark(loc) << "pset => " << Out.str();
  };

  // Do not emit more than one diagonistic for the same task deref location.
  // Since cowait hides a bunch of logic and calls to the promise type, just
  // have one per suspend expr.
  if (tasks.count(addr) && emittedDanglingTasks.count(loc))
    return;

  bool psetRemarkEmitted = false;
  if (opts.emitRemarkPsetAlways()) {
    emitPsetRemark();
    psetRemarkEmitted = true;
  }

  // 2.4.2 - On every dereference of a Pointer p, enforce that p is valid.
  if (!hasInvalid && !hasNullptr)
    return;

  // Looks like we found a bad path leading to this deference point,
  // diagnose it.
  StringRef varName = getVarNameFromValue(addr);
  auto D = emitWarning(loc);

  if (tasks.count(addr))
    D << "use of coroutine '" << varName << "' with dangling reference";
  else if (forRetLambda)
    D << "returned lambda captures local variable";
  else
    D << "use of invalid pointer '" << varName << "'";

  if (hasInvalid && opts.emitHistoryInvalid())
    emitInvalidHistory(D, addr, loc, forRetLambda);

  if (hasNullptr && opts.emitHistoryNull()) {
    assert(pmapNullHist.count(addr) && "expected nullptr hist");
    auto &note = pmapNullHist[addr];
    D.attachNote(*note) << "invalidated here";
  }

  if (!psetRemarkEmitted && opts.emitRemarkPsetInvalid())
    emitPsetRemark();
}

static FuncOp getCalleeFromSymbol(ModuleOp mod, StringRef name) {
  auto global = mlir::SymbolTable::lookupSymbolIn(mod, name);
  assert(global && "expected to find symbol for function");
  return dyn_cast<FuncOp>(global);
}

static const clang::CXXMethodDecl *getMethod(ModuleOp mod, StringRef name) {
  auto method = getCalleeFromSymbol(mod, name);
  if (!method || method.getBuiltin())
    return nullptr;
  return dyn_cast<clang::CXXMethodDecl>(method.getAstAttr().getAstDecl());
}

void LifetimeCheckPass::checkMoveAssignment(CallOp callOp,
                                            const clang::CXXMethodDecl *m) {
  // MyPointer::operator=(MyPointer&&)(%dst, %src)
  // or
  // MyOwner::operator=(MyOwner&&)(%dst, %src)
  auto dst = callOp.getOperand(0);
  auto src = callOp.getOperand(1);

  // Move assignments between pointer categories.
  if (ptrs.count(dst) && ptrs.count(src)) {
    // Note that the current pattern here usually comes from a xvalue in src
    // where all the initialization is done, and this move assignment is
    // where we finally materialize it back to the original pointer category.
    getPmap()[dst] = getPmap()[src];

    // 2.4.2 - It is an error to use a moved-from object.
    // To that intent we mark src's pset with invalid.
    markPsetInvalid(src, InvalidStyle::MovedFrom, callOp.getLoc());
    return;
  }

  // Copy assignments between pointer categories.
  if (owners.count(dst) && owners.count(src)) {
    // Handle as a non const use of owner, invalidating pointers.
    checkNonConstUseOfOwner(dst, callOp.getLoc());

    // 2.4.2 - It is an error to use a moved-from object.
    // To that intent we mark src's pset with invalid.
    markPsetInvalid(src, InvalidStyle::MovedFrom, callOp.getLoc());
  }
}

void LifetimeCheckPass::checkCopyAssignment(CallOp callOp,
                                            const clang::CXXMethodDecl *m) {
  // MyIntOwner::operator=(MyIntOwner&)(%dst, %src)
  auto dst = callOp.getOperand(0);
  auto src = callOp.getOperand(1);

  // Copy assignment between owner categories.
  if (owners.count(dst) && owners.count(src))
    return checkNonConstUseOfOwner(dst, callOp.getLoc());

  // Copy assignment between pointer categories.
  if (ptrs.count(dst) && ptrs.count(src)) {
    getPmap()[dst] = getPmap()[src];
    return;
  }
}

// User defined ctors that initialize from owner types is one
// way of tracking owned pointers.
//
// Example:
//  MyIntPointer::MyIntPointer(MyIntOwner const&)(%5, %4)
//
bool LifetimeCheckPass::isCtorInitPointerFromOwner(
    CallOp callOp, const clang::CXXConstructorDecl *ctor) {
  if (callOp.getNumOperands() < 2)
    return false;

  // FIXME: should we scan all arguments past first to look for an owner?
  auto addr = callOp.getOperand(0);
  auto owner = callOp.getOperand(1);

  if (ptrs.count(addr) && owners.count(owner))
    return true;

  return false;
}

void LifetimeCheckPass::checkCtor(CallOp callOp,
                                  const clang::CXXConstructorDecl *ctor) {
  // TODO: zero init
  // 2.4.2 if the initialization is default initialization or zero
  // initialization, example:
  //
  //    int* p{};
  //    string_view p;
  //
  // both results in pset(p) == {null}
  if (ctor->isDefaultConstructor()) {
    // First argument passed is always the alloca for the 'this' ptr.
    auto addr = callOp.getOperand(0);

    // Currently two possible actions:
    // 1. Skip Owner category initialization.
    // 2. Initialize Pointer categories.
    if (owners.count(addr))
      return;

    if (!ptrs.count(addr))
      return;

    // Not interested in block/function arguments or any indirect
    // provided alloca address.
    if (!dyn_cast_or_null<AllocaOp>(addr.getDefiningOp()))
      return;

    getPmap()[addr].clear();
    getPmap()[addr].insert(State::getNullPtr());
    pmapNullHist[addr] = callOp.getLoc();
    return;
  }

  // User defined copy ctor calls ...
  if (ctor->isCopyConstructor()) {
    llvm_unreachable("NYI");
  }

  if (isCtorInitPointerFromOwner(callOp, ctor)) {
    auto addr = callOp.getOperand(0);
    auto owner = callOp.getOperand(1);
    getPmap()[addr].clear();
    getPmap()[addr].insert(State::getOwnedBy(owner));
    return;
  }
}

void LifetimeCheckPass::checkOperators(CallOp callOp,
                                       const clang::CXXMethodDecl *m) {
  auto addr = callOp.getOperand(0);
  if (owners.count(addr)) {
    // const access to the owner is fine.
    if (m->isConst())
      return;
    // TODO: this is a place where we can hook in some idiom recocgnition
    // so we don't need to use actual source code annotation to make assumptions
    // on methods we understand and know to behave nicely.
    //
    // In P1179, section 2.5.7.12, the use of [[gsl::lifetime_const]] is
    // suggested, but it's not part of clang (will it ever?)
    return checkNonConstUseOfOwner(addr, callOp.getLoc());
  }

  if (ptrs.count(addr)) {
    // The assumption is that method calls on pointer types should trigger
    // deref checking.
    checkPointerDeref(addr, callOp.getLoc());
  }

  // FIXME: we also need to look at operators from non owner or pointer
  // types that could be using Owner/Pointer types as parameters.
}

bool LifetimeCheckPass::isNonConstUseOfOwner(CallOp callOp,
                                             const clang::CXXMethodDecl *m) {
  if (m->isConst())
    return false;
  auto addr = callOp.getOperand(0);
  if (owners.count(addr))
    return true;
  return false;
}

void LifetimeCheckPass::checkNonConstUseOfOwner(mlir::Value ownerAddr,
                                                mlir::Location loc) {
  // 2.4.2 - On every non-const use of a local Owner o:
  //
  // - For each entry e in pset(s): Remove e from pset(s), and if no other
  // Owner’s pset contains only e, then KILL(e).
  kill(State::getOwnedBy(ownerAddr), InvalidStyle::NonConstUseOfOwner, loc);

  // - Set pset(o) = {o__N'}, where N is one higher than the highest
  // previously used suffix. For example, initially pset(o) is {o__1'}, on
  // o’s first non-const use pset(o) becomes {o__2'}, on o’s second non-const
  // use pset(o) becomes {o__3'}, and so on.
  incOwner(ownerAddr);
  return;
}

void LifetimeCheckPass::checkForOwnerAndPointerArguments(CallOp callOp,
                                                         unsigned firstArgIdx) {
  auto numOperands = callOp.getNumOperands();
  if (firstArgIdx >= numOperands)
    return;

  llvm::SmallSetVector<mlir::Value, 8> ownersToInvalidate, ptrsToDeref;
  for (unsigned i = firstArgIdx, e = numOperands; i != e; ++i) {
    auto arg = callOp.getOperand(i);
    // FIXME: apply p1179 rules as described in 2.5. Very conservative for now:
    //
    // - Owners: always invalidate.
    // - Pointers: always check for deref.
    // - Coroutine tasks: check the task for deref when calling methods of
    //   the task, but also when the passing the task around to other functions.
    //
    // FIXME: even before 2.5 we should only invalidate non-const param types.
    if (owners.count(arg))
      ownersToInvalidate.insert(arg);
    if (ptrs.count(arg))
      ptrsToDeref.insert(arg);
    if (tasks.count(arg))
      ptrsToDeref.insert(arg);
  }

  // FIXME: CIR should track source info on the passed args, so we can get
  // accurate location for why the invalidation happens.
  for (auto o : ownersToInvalidate)
    checkNonConstUseOfOwner(o, callOp.getLoc());
  for (auto p : ptrsToDeref)
    checkPointerDeref(p, callOp.getLoc());
}

void LifetimeCheckPass::checkOtherMethodsAndFunctions(
    CallOp callOp, const clang::CXXMethodDecl *m) {
  unsigned firstArgIdx = 0;

  // Looks at a method 'this' pointer:
  // - If a method call to a class we consider interesting, like a method
  //   call on a coroutine task (promise_type).
  // - Skip the 'this' for any other method.
  if (m && !tasks.count(callOp.getOperand(firstArgIdx)))
    firstArgIdx++;
  checkForOwnerAndPointerArguments(callOp, firstArgIdx);
}

bool LifetimeCheckPass::isOwnerOrPointerClassMethod(
    mlir::Value firstParam, const clang::CXXMethodDecl *m) {
  // For the sake of analysis, these behave like regular functions
  if (!m || m->isStatic())
    return false;
  if (owners.count(firstParam) || ptrs.count(firstParam))
    return true;
  return false;
}

bool LifetimeCheckPass::isLambdaType(mlir::Type ty) {
  if (IsLambdaTyCache.count(ty))
    return IsLambdaTyCache[ty];

  IsLambdaTyCache[ty] = false;
  auto taskTy = ty.dyn_cast<mlir::cir::StructType>();
  if (!taskTy)
    return false;
  auto recordDecl = taskTy.getAst()->getAstDecl();
  if (recordDecl->isLambda())
    IsLambdaTyCache[ty] = true;

  return IsLambdaTyCache[ty];
}

bool LifetimeCheckPass::isTaskType(mlir::Value taskVal) {
  auto ty = taskVal.getType();
  if (IsTaskTyCache.count(ty))
    return IsTaskTyCache[ty];

  IsTaskTyCache[ty] = false;
  auto taskTy = taskVal.getType().dyn_cast<mlir::cir::StructType>();
  if (!taskTy)
    return false;
  auto recordDecl = taskTy.getAst()->getAstDecl();
  auto *spec = dyn_cast<clang::ClassTemplateSpecializationDecl>(recordDecl);
  if (!spec)
    return false;

  for (auto *sub : spec->decls()) {
    auto *subRec = dyn_cast<clang::CXXRecordDecl>(sub);
    if (subRec && subRec->getDeclName().isIdentifier() &&
        subRec->getName() == "promise_type") {
      IsTaskTyCache[ty] = true;
      break;
    }
  }

  return IsTaskTyCache[ty];
}

void LifetimeCheckPass::trackCallToCoroutine(CallOp callOp) {
  if (auto fnName = callOp.getCallee()) {
    auto calleeFuncOp = getCalleeFromSymbol(theModule, *fnName);
    if (calleeFuncOp &&
        (calleeFuncOp.getCoroutine() ||
         (calleeFuncOp.isDeclaration() && callOp->getNumResults() > 0 &&
          isTaskType(callOp->getResult(0))))) {
      currScope->localTempTasks.insert(callOp->getResult(0));
    }
    return;
  }
  // Handle indirect calls to coroutines, for instance when
  // lambda coroutines are involved with invokers.
  if (callOp->getNumResults() > 0 && isTaskType(callOp->getResult(0))) {
    // FIXME: get more guarantees to prevent false positives (perhaps
    // apply some tracking analysis before this pass and check for lambda
    // idioms).
    currScope->localTempTasks.insert(callOp->getResult(0));
  }
}

void LifetimeCheckPass::checkCall(CallOp callOp) {
  if (callOp.getNumOperands() == 0)
    return;

  // Identify calls to coroutines and track returning temporary task types.
  //
  // Note that we can't reliably know if a function is a coroutine only as
  // part of declaration
  trackCallToCoroutine(callOp);

  // FIXME: General indirect calls not yet supported.
  if (!callOp.getCallee())
    return;

  auto fnName = *callOp.getCallee();
  auto methodDecl = getMethod(theModule, fnName);
  if (!isOwnerOrPointerClassMethod(callOp.getOperand(0), methodDecl))
    return checkOtherMethodsAndFunctions(callOp, methodDecl);

  // From this point on only owner and pointer class methods handling,
  // starting from special methods.
  if (const auto *ctor = dyn_cast<clang::CXXConstructorDecl>(methodDecl))
    return checkCtor(callOp, ctor);
  if (methodDecl->isMoveAssignmentOperator())
    return checkMoveAssignment(callOp, methodDecl);
  if (methodDecl->isCopyAssignmentOperator())
    return checkCopyAssignment(callOp, methodDecl);
  if (methodDecl->isOverloadedOperator())
    return checkOperators(callOp, methodDecl);

  // For any other methods...

  // Non-const member call to a Owner invalidates any of its users.
  if (isNonConstUseOfOwner(callOp, methodDecl))
    return checkNonConstUseOfOwner(callOp.getOperand(0), callOp.getLoc());

  // Take a pset(Ptr) = { Ownr' } where Own got invalidated, this will become
  // invalid access to Ptr if any of its methods are used.
  auto addr = callOp.getOperand(0);
  if (ptrs.count(addr))
    return checkPointerDeref(addr, callOp.getLoc());
}

void LifetimeCheckPass::checkOperation(Operation *op) {
  if (isa<::mlir::ModuleOp>(op)) {
    theModule = cast<::mlir::ModuleOp>(op);
    for (Region &region : op->getRegions())
      checkRegion(region);
    return;
  }

  if (isa<ScopeOp>(op)) {
    // Add a new scope. Note that as part of the scope cleanup process
    // we apply section 2.3 KILL(x) functionality, turning relevant
    // references invalid.
    //
    // No need to create a new pmap when entering a new scope since it
    // doesn't cause control flow to diverge (as it does in presence
    // of cir::IfOp or cir::SwitchOp).
    //
    // Also note that for dangling pointers coming from if init stmts
    // should be caught just fine, given that a ScopeOp embraces a IfOp.
    LexicalScopeContext lexScope{op};
    LexicalScopeGuard scopeGuard{*this, &lexScope};
    for (Region &region : op->getRegions())
      checkRegion(region);
    return;
  }

  // FIXME: we can do better than sequence of dyn_casts.
  if (auto fnOp = dyn_cast<cir::FuncOp>(op))
    return checkFunc(fnOp);
  if (auto ifOp = dyn_cast<IfOp>(op))
    return checkIf(ifOp);
  if (auto switchOp = dyn_cast<SwitchOp>(op))
    return checkSwitch(switchOp);
  if (auto loopOp = dyn_cast<LoopOp>(op))
    return checkLoop(loopOp);
  if (auto allocaOp = dyn_cast<AllocaOp>(op))
    return checkAlloca(allocaOp);
  if (auto storeOp = dyn_cast<StoreOp>(op))
    return checkStore(storeOp);
  if (auto loadOp = dyn_cast<LoadOp>(op))
    return checkLoad(loadOp);
  if (auto callOp = dyn_cast<CallOp>(op))
    return checkCall(callOp);
  if (auto awaitOp = dyn_cast<AwaitOp>(op))
    return checkAwait(awaitOp);
  if (auto returnOp = dyn_cast<ReturnOp>(op))
    return checkReturn(returnOp);
}

void LifetimeCheckPass::runOnOperation() {
  opts.parseOptions(*this);
  Operation *op = getOperation();
  checkOperation(op);
}

std::unique_ptr<Pass> mlir::createLifetimeCheckPass() {
  // FIXME: MLIR requres a default "constructor", but should never
  // be used.
  llvm_unreachable("Check requires clang::ASTContext, use the other ctor");
  return std::make_unique<LifetimeCheckPass>();
}

std::unique_ptr<Pass> mlir::createLifetimeCheckPass(clang::ASTContext *astCtx) {
  auto lifetime = std::make_unique<LifetimeCheckPass>();
  lifetime->setASTContext(astCtx);
  return std::move(lifetime);
}

std::unique_ptr<Pass> mlir::createLifetimeCheckPass(ArrayRef<StringRef> remark,
                                                    ArrayRef<StringRef> hist,
                                                    unsigned hist_limit,
                                                    clang::ASTContext *astCtx) {
  auto lifetime = std::make_unique<LifetimeCheckPass>();
  lifetime->setASTContext(astCtx);
  lifetime->opts.parseOptions(remark, hist, hist_limit);
  return std::move(lifetime);
}

//===----------------------------------------------------------------------===//
// Dump & print helpers
//===----------------------------------------------------------------------===//

void LifetimeCheckPass::LexicalScopeContext::dumpLocalValues() {
  llvm::errs() << "Local values: { ";
  for (auto value : localValues) {
    llvm::errs() << getVarNameFromValue(value);
    llvm::errs() << ", ";
  }
  llvm::errs() << "}\n";
}

void LifetimeCheckPass::State::dump(llvm::raw_ostream &OS, int ownedGen) {
  switch (val.getInt()) {
  case Invalid:
    OS << "invalid";
    break;
  case NullPtr:
    OS << "nullptr";
    break;
  case Global:
    OS << "global";
    break;
  case LocalValue:
    OS << getVarNameFromValue(val.getPointer());
    break;
  case OwnedBy:
    ownedGen++; // Start from 1.
    OS << getVarNameFromValue(val.getPointer()) << "__" << ownedGen << "'";
    break;
  default:
    llvm_unreachable("Not handled");
  }
}

void LifetimeCheckPass::printPset(PSetType &pset, llvm::raw_ostream &OS) {
  OS << "{ ";
  auto size = pset.size();
  for (auto s : pset) {
    int ownerGen = 0;
    if (s.isOwnedBy())
      ownerGen = owners[s.getData()];
    s.dump(OS, ownerGen);
    size--;
    if (size > 0)
      OS << ", ";
  }
  OS << " }";
}

void LifetimeCheckPass::dumpCurrentPmap() { dumpPmap(*currPmap); }

void LifetimeCheckPass::dumpPmap(PMapType &pmap) {
  llvm::errs() << "pmap {\n";
  int entry = 0;
  for (auto &mapEntry : pmap) {
    llvm::errs() << "  " << entry << ": " << getVarNameFromValue(mapEntry.first)
                 << "  "
                 << "=> ";
    printPset(mapEntry.second);
    llvm::errs() << "\n";
    entry++;
  }
  llvm::errs() << "}\n";
}
