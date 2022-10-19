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
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace cir;

namespace {
struct LifetimeCheckPass : public LifetimeCheckBase<LifetimeCheckPass> {
  LifetimeCheckPass() = default;
  void runOnOperation() override;

  void checkOperation(Operation *op);
  void checkFunc(Operation *op);
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

  void checkPointerDeref(mlir::Value addr, mlir::Location loc);

  void checkCtor(CallOp callOp, const clang::CXXConstructorDecl *ctor);
  void checkMoveAssignment(CallOp callOp, const clang::CXXMethodDecl *m);
  void checkOperatorStar(CallOp callOp);

  // Tracks current module.
  ModuleOp theModule;

  // Common helpers.
  bool isCtorInitPointerFromOwner(CallOp callOp,
                                  const clang::CXXConstructorDecl *ctor);
  bool isNonConstUseOfOwner(CallOp callOp, const clang::CXXMethodDecl *m);

  // Diagnostic helpers.
  void emitInvalidHistory(mlir::InFlightDiagnostic &D, mlir::Value histKey);

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

    void parseOptions(LifetimeCheckPass &pass) {
      for (auto &remark : pass.remarksList) {
        val |= StringSwitch<unsigned>(remark)
                   .Case("pset-invalid", RemarkPsetInvalid)
                   .Case("pset-always", RemarkPsetAlways)
                   .Case("all", RemarkAll)
                   .Default(None);
      }
      for (auto &h : pass.historyList) {
        val |= StringSwitch<unsigned>(h)
                   .Case("invalid", HistoryInvalid)
                   .Case("null", HistoryNull)
                   .Case("all", HistoryAll)
                   .Default(None);
      }
      histLimit = pass.historyLimit;
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
      if (val.getInt() == LocalValue && RHS.val.getInt() == LocalValue)
        return val.getPointer().getAsOpaquePointer() <
               RHS.val.getPointer().getAsOpaquePointer();
      return val.getInt() < RHS.val.getInt();
    }
    bool operator==(const State &RHS) const {
      if (val.getInt() == LocalValue && RHS.val.getInt() == LocalValue)
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

  // Useful helpers for debugging
  void printPset(PSetType &pset, llvm::raw_ostream &OS = llvm::errs());
  LLVM_DUMP_METHOD void dumpPmap(PMapType &pmap);
  LLVM_DUMP_METHOD void dumpCurrentPmap();

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
    llvm::SmallVector<mlir::Value, 4> localValues;

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

static Location getEndLocForHist(Region *R) {
  auto ifOp = dyn_cast<IfOp>(R->getParentOp());
  assert(ifOp && "what other regions create their own scope?");
  if (&ifOp.getThenRegion() == R)
    return getEndLoc(ifOp.getLoc());
  return getEndLoc(ifOp.getLoc(), /*idx=*/3);
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
    getPmap().erase(v);
  }
}

void LifetimeCheckPass::LexicalScopeGuard::cleanup() {
  auto *localScope = Pass.currScope;
  // If we are cleaning up at the function level, nothing
  // to do here cause we are past all possible deference points
  if (localScope->Depth == 0)
    return;

  for (auto pointee : localScope->localValues)
    Pass.kill(State::getLocalValue(pointee), InvalidStyle::EndOfScope,
              getEndLocForHist(*localScope));
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

void LifetimeCheckPass::checkFunc(Operation *op) {
  // FIXME: perhaps this should be a function pass, but for now make
  // sure we reset the state before looking at other functions.
  if (currPmap)
    getPmap().clear();
  pmapNullHist.clear();
  invalidHist.clear();

  // Add a new scope. Note that as part of the scope cleanup process
  // we apply section 2.3 KILL(x) functionality, turning relevant
  // references invalid.
  LexicalScopeContext lexScope{op};
  LexicalScopeGuard scopeGuard{*this, &lexScope};

  // Create a new pmap for this function.
  PMapType localPmap{};
  PmapGuard pmapGuard{*this, &localPmap};

  for (Region &region : op->getRegions())
    checkRegion(region);

  // FIXME: store the pmap result for this function, we
  // could do some interesting IPA stuff using this info.
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
    currScope->localValues.push_back(addr);
    break;
  case TypeCategory::Value: {
    // 2.4.2 - When a local Value x is declared, add (x, {x}) to pmap.
    getPmap()[addr].insert(State::getLocalValue(addr));
    currScope->localValues.push_back(addr);
    return;
  }
  default:
    llvm_unreachable("NYI");
  }

  // If other styles of initialization gets added, required to add support
  // here.
  auto varDecl = allocaOp.getAst();
  assert(!varDecl ||
         (!allocaOp.getInit() || !varDecl->getAstDecl()->isDirectInit()) &&
             "not implemented");
}

void LifetimeCheckPass::checkStore(StoreOp storeOp) {
  auto addr = storeOp.getAddr();

  // We only care about stores that change local pointers, local values
  // are not interesting here (just yet).
  if (!ptrs.count(addr))
    return;

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
                                           mlir::Value histKey) {
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
      StringRef outOfScopeVarName = getVarNameFromValue(*info.val);
      D.attachNote(info.loc) << "pointee '" << outOfScopeVarName
                             << "' invalidated at end of scope";
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

void LifetimeCheckPass::checkPointerDeref(mlir::Value addr,
                                          mlir::Location loc) {
  bool hasInvalid = getPmap()[addr].count(State::getInvalid());
  bool hasNullptr = getPmap()[addr].count(State::getNullPtr());

  auto emitPsetRemark = [&] {
    llvm::SmallString<128> psetStr;
    llvm::raw_svector_ostream Out(psetStr);
    printPset(getPmap()[addr], Out);
    emitRemark(loc) << "pset => " << Out.str();
  };

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
  D << "use of invalid pointer '" << varName << "'";

  if (hasInvalid && opts.emitHistoryInvalid())
    emitInvalidHistory(D, addr);

  if (hasNullptr && opts.emitHistoryNull()) {
    assert(pmapNullHist.count(addr) && "expected nullptr hist");
    auto &note = pmapNullHist[addr];
    D.attachNote(*note) << "invalidated here";
  }

  if (!psetRemarkEmitted && opts.emitRemarkPsetInvalid())
    emitPsetRemark();
}

const clang::CXXMethodDecl *getMethod(ModuleOp mod, StringRef name) {
  auto *global = mlir::SymbolTable::lookupSymbolIn(mod, name);
  assert(global && "expected to find symbol");
  auto method = dyn_cast<FuncOp>(global);
  if (!method)
    return nullptr;
  return dyn_cast<clang::CXXMethodDecl>(method.getAstAttr().getAstDecl());
}

void LifetimeCheckPass::checkMoveAssignment(CallOp callOp,
                                            const clang::CXXMethodDecl *m) {
  // MyIntPointer::operator=(MyIntPointer&&)(%dst, %src)
  auto dst = callOp.getOperand(0);
  auto src = callOp.getOperand(1);

  // Currently only handle move assignments between pointer categories.
  if (!(ptrs.count(dst) && ptrs.count(src)))
    return;

  // Note that the current pattern here usually comes from a xvalue in src
  // where all the initialization is done, and this move assignment is
  // where we finally materialize it back to the original pointer category.
  // TODO: should CIR ops retain xvalue information somehow?
  getPmap()[dst] = getPmap()[src];
  // TODO: should this be null? or should we swap dst/src pset state?
  // For now just consider moved-from state as invalid.
  getPmap()[src].clear();
  getPmap()[src].insert(State::getInvalid());
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

static bool isOperatorStar(const clang::CXXMethodDecl *m) {
  if (!m->isOverloadedOperator())
    return false;
  return m->getOverloadedOperator() == clang::OverloadedOperatorKind::OO_Star;
}

static bool sinkUnsupportedOperator(const clang::CXXMethodDecl *m) {
  if (!m->isOverloadedOperator())
    return false;
  if (!isOperatorStar(m))
    llvm_unreachable("NYI");
  return false;
}

void LifetimeCheckPass::checkOperatorStar(CallOp callOp) {
  auto addr = callOp.getOperand(0);
  if (!ptrs.count(addr))
    return;

  checkPointerDeref(addr, callOp.getLoc());
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

void LifetimeCheckPass::checkCall(CallOp callOp) {
  if (callOp.getNumOperands() == 0)
    return;

  const auto *methodDecl = getMethod(theModule, callOp.getCallee());
  if (!methodDecl)
    return;

  if (const auto *ctor = dyn_cast<clang::CXXConstructorDecl>(methodDecl))
    return checkCtor(callOp, ctor);
  if (methodDecl->isMoveAssignmentOperator())
    return checkMoveAssignment(callOp, methodDecl);
  if (methodDecl->isCopyAssignmentOperator())
    llvm_unreachable("NYI");
  if (isOperatorStar(methodDecl))
    return checkOperatorStar(callOp);
  if (sinkUnsupportedOperator(methodDecl))
    return;

  // For any other methods...

  // Non-const member call to a Owner invalidates any of its users.
  if (isNonConstUseOfOwner(callOp, methodDecl)) {
    auto ownerAddr = callOp.getOperand(0);
    // 2.4.2 - On every non-const use of a local Owner o:
    //
    // - For each entry e in pset(s): Remove e from pset(s), and if no other
    // Owner’s pset contains only e, then KILL(e).
    kill(State::getOwnedBy(ownerAddr), InvalidStyle::NonConstUseOfOwner,
         callOp.getLoc());

    // - Set pset(o) = {o__N'}, where N is one higher than the highest
    // previously used suffix. For example, initially pset(o) is {o__1'}, on
    // o’s first non-const use pset(o) becomes {o__2'}, on o’s second non-const
    // use pset(o) becomes {o__3'}, and so on.
    incOwner(ownerAddr);
    return;
  }

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

  if (isa<cir::FuncOp>(op))
    return checkFunc(op);
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
