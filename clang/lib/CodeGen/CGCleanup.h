//===-- CGCleanup.h - Classes for cleanups IR generation --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes support the generation of LLVM IR for cleanups.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGCLEANUP_H
#define CLANG_CODEGEN_CGCLEANUP_H

/// EHScopeStack is defined in CodeGenFunction.h, but its
/// implementation is in this file and in CGCleanup.cpp.
#include "CodeGenFunction.h"

namespace llvm {
  class Value;
  class BasicBlock;
}

namespace clang {
namespace CodeGen {

/// A protected scope for zero-cost EH handling.
class EHScope {
  llvm::BasicBlock *CachedLandingPad;
  llvm::BasicBlock *CachedEHDispatchBlock;

  EHScopeStack::stable_iterator EnclosingEHScope;

  class CommonBitFields {
    friend class EHScope;
    unsigned Kind : 2;
  };
  enum { NumCommonBits = 2 };

protected:
  class CatchBitFields {
    friend class EHCatchScope;
    unsigned : NumCommonBits;

    unsigned NumHandlers : 32 - NumCommonBits;
  };

  class CleanupBitFields {
    friend class EHCleanupScope;
    unsigned : NumCommonBits;

    /// Whether this cleanup needs to be run along normal edges.
    unsigned IsNormalCleanup : 1;

    /// Whether this cleanup needs to be run along exception edges.
    unsigned IsEHCleanup : 1;

    /// Whether this cleanup is currently active.
    unsigned IsActive : 1;

    /// Whether the normal cleanup should test the activation flag.
    unsigned TestFlagInNormalCleanup : 1;

    /// Whether the EH cleanup should test the activation flag.
    unsigned TestFlagInEHCleanup : 1;

    /// The amount of extra storage needed by the Cleanup.
    /// Always a multiple of the scope-stack alignment.
    unsigned CleanupSize : 12;

    /// The number of fixups required by enclosing scopes (not including
    /// this one).  If this is the top cleanup scope, all the fixups
    /// from this index onwards belong to this scope.
    unsigned FixupDepth : 32 - 17 - NumCommonBits; // currently 13    
  };

  class FilterBitFields {
    friend class EHFilterScope;
    unsigned : NumCommonBits;

    unsigned NumFilters : 32 - NumCommonBits;
  };

  union {
    CommonBitFields CommonBits;
    CatchBitFields CatchBits;
    CleanupBitFields CleanupBits;
    FilterBitFields FilterBits;
  };

public:
  enum Kind { Cleanup, Catch, Terminate, Filter };

  EHScope(Kind kind, EHScopeStack::stable_iterator enclosingEHScope)
    : CachedLandingPad(0), CachedEHDispatchBlock(0),
      EnclosingEHScope(enclosingEHScope) {
    CommonBits.Kind = kind;
  }

  Kind getKind() const { return static_cast<Kind>(CommonBits.Kind); }

  llvm::BasicBlock *getCachedLandingPad() const {
    return CachedLandingPad;
  }

  void setCachedLandingPad(llvm::BasicBlock *block) {
    CachedLandingPad = block;
  }

  llvm::BasicBlock *getCachedEHDispatchBlock() const {
    return CachedEHDispatchBlock;
  }

  void setCachedEHDispatchBlock(llvm::BasicBlock *block) {
    CachedEHDispatchBlock = block;
  }

  bool hasEHBranches() const {
    if (llvm::BasicBlock *block = getCachedEHDispatchBlock())
      return !block->use_empty();
    return false;
  }

  EHScopeStack::stable_iterator getEnclosingEHScope() const {
    return EnclosingEHScope;
  }
};

/// A scope which attempts to handle some, possibly all, types of
/// exceptions.
///
/// Objective C \@finally blocks are represented using a cleanup scope
/// after the catch scope.
class EHCatchScope : public EHScope {
  // In effect, we have a flexible array member
  //   Handler Handlers[0];
  // But that's only standard in C99, not C++, so we have to do
  // annoying pointer arithmetic instead.

public:
  struct Handler {
    /// A type info value, or null (C++ null, not an LLVM null pointer)
    /// for a catch-all.
    llvm::Value *Type;

    /// The catch handler for this type.
    llvm::BasicBlock *Block;

    bool isCatchAll() const { return Type == 0; }
  };

private:
  friend class EHScopeStack;

  Handler *getHandlers() {
    return reinterpret_cast<Handler*>(this+1);
  }

  const Handler *getHandlers() const {
    return reinterpret_cast<const Handler*>(this+1);
  }

public:
  static size_t getSizeForNumHandlers(unsigned N) {
    return sizeof(EHCatchScope) + N * sizeof(Handler);
  }

  EHCatchScope(unsigned numHandlers,
               EHScopeStack::stable_iterator enclosingEHScope)
    : EHScope(Catch, enclosingEHScope) {
    CatchBits.NumHandlers = numHandlers;
  }

  unsigned getNumHandlers() const {
    return CatchBits.NumHandlers;
  }

  void setCatchAllHandler(unsigned I, llvm::BasicBlock *Block) {
    setHandler(I, /*catchall*/ 0, Block);
  }

  void setHandler(unsigned I, llvm::Value *Type, llvm::BasicBlock *Block) {
    assert(I < getNumHandlers());
    getHandlers()[I].Type = Type;
    getHandlers()[I].Block = Block;
  }

  const Handler &getHandler(unsigned I) const {
    assert(I < getNumHandlers());
    return getHandlers()[I];
  }

  typedef const Handler *iterator;
  iterator begin() const { return getHandlers(); }
  iterator end() const { return getHandlers() + getNumHandlers(); }

  static bool classof(const EHScope *Scope) {
    return Scope->getKind() == Catch;
  }
};

/// A cleanup scope which generates the cleanup blocks lazily.
class EHCleanupScope : public EHScope {
  /// The nearest normal cleanup scope enclosing this one.
  EHScopeStack::stable_iterator EnclosingNormal;

  /// The nearest EH scope enclosing this one.
  EHScopeStack::stable_iterator EnclosingEH;

  /// The dual entry/exit block along the normal edge.  This is lazily
  /// created if needed before the cleanup is popped.
  llvm::BasicBlock *NormalBlock;

  /// An optional i1 variable indicating whether this cleanup has been
  /// activated yet.
  llvm::AllocaInst *ActiveFlag;

  /// Extra information required for cleanups that have resolved
  /// branches through them.  This has to be allocated on the side
  /// because everything on the cleanup stack has be trivially
  /// movable.
  struct ExtInfo {
    /// The destinations of normal branch-afters and branch-throughs.
    llvm::SmallPtrSet<llvm::BasicBlock*, 4> Branches;

    /// Normal branch-afters.
    SmallVector<std::pair<llvm::BasicBlock*,llvm::ConstantInt*>, 4>
      BranchAfters;
  };
  mutable struct ExtInfo *ExtInfo;

  struct ExtInfo &getExtInfo() {
    if (!ExtInfo) ExtInfo = new struct ExtInfo();
    return *ExtInfo;
  }

  const struct ExtInfo &getExtInfo() const {
    if (!ExtInfo) ExtInfo = new struct ExtInfo();
    return *ExtInfo;
  }

public:
  /// Gets the size required for a lazy cleanup scope with the given
  /// cleanup-data requirements.
  static size_t getSizeForCleanupSize(size_t Size) {
    return sizeof(EHCleanupScope) + Size;
  }

  size_t getAllocatedSize() const {
    return sizeof(EHCleanupScope) + CleanupBits.CleanupSize;
  }

  EHCleanupScope(bool isNormal, bool isEH, bool isActive,
                 unsigned cleanupSize, unsigned fixupDepth,
                 EHScopeStack::stable_iterator enclosingNormal,
                 EHScopeStack::stable_iterator enclosingEH)
    : EHScope(EHScope::Cleanup, enclosingEH), EnclosingNormal(enclosingNormal),
      NormalBlock(0), ActiveFlag(0), ExtInfo(0) {
    CleanupBits.IsNormalCleanup = isNormal;
    CleanupBits.IsEHCleanup = isEH;
    CleanupBits.IsActive = isActive;
    CleanupBits.TestFlagInNormalCleanup = false;
    CleanupBits.TestFlagInEHCleanup = false;
    CleanupBits.CleanupSize = cleanupSize;
    CleanupBits.FixupDepth = fixupDepth;

    assert(CleanupBits.CleanupSize == cleanupSize && "cleanup size overflow");
  }

  ~EHCleanupScope() {
    delete ExtInfo;
  }

  bool isNormalCleanup() const { return CleanupBits.IsNormalCleanup; }
  llvm::BasicBlock *getNormalBlock() const { return NormalBlock; }
  void setNormalBlock(llvm::BasicBlock *BB) { NormalBlock = BB; }

  bool isEHCleanup() const { return CleanupBits.IsEHCleanup; }
  llvm::BasicBlock *getEHBlock() const { return getCachedEHDispatchBlock(); }
  void setEHBlock(llvm::BasicBlock *BB) { setCachedEHDispatchBlock(BB); }

  bool isActive() const { return CleanupBits.IsActive; }
  void setActive(bool A) { CleanupBits.IsActive = A; }

  llvm::AllocaInst *getActiveFlag() const { return ActiveFlag; }
  void setActiveFlag(llvm::AllocaInst *Var) { ActiveFlag = Var; }

  void setTestFlagInNormalCleanup() {
    CleanupBits.TestFlagInNormalCleanup = true;
  }
  bool shouldTestFlagInNormalCleanup() const {
    return CleanupBits.TestFlagInNormalCleanup;
  }

  void setTestFlagInEHCleanup() {
    CleanupBits.TestFlagInEHCleanup = true;
  }
  bool shouldTestFlagInEHCleanup() const {
    return CleanupBits.TestFlagInEHCleanup;
  }

  unsigned getFixupDepth() const { return CleanupBits.FixupDepth; }
  EHScopeStack::stable_iterator getEnclosingNormalCleanup() const {
    return EnclosingNormal;
  }

  size_t getCleanupSize() const { return CleanupBits.CleanupSize; }
  void *getCleanupBuffer() { return this + 1; }

  EHScopeStack::Cleanup *getCleanup() {
    return reinterpret_cast<EHScopeStack::Cleanup*>(getCleanupBuffer());
  }

  /// True if this cleanup scope has any branch-afters or branch-throughs.
  bool hasBranches() const { return ExtInfo && !ExtInfo->Branches.empty(); }

  /// Add a branch-after to this cleanup scope.  A branch-after is a
  /// branch from a point protected by this (normal) cleanup to a
  /// point in the normal cleanup scope immediately containing it.
  /// For example,
  ///   for (;;) { A a; break; }
  /// contains a branch-after.
  ///
  /// Branch-afters each have their own destination out of the
  /// cleanup, guaranteed distinct from anything else threaded through
  /// it.  Therefore branch-afters usually force a switch after the
  /// cleanup.
  void addBranchAfter(llvm::ConstantInt *Index,
                      llvm::BasicBlock *Block) {
    struct ExtInfo &ExtInfo = getExtInfo();
    if (ExtInfo.Branches.insert(Block))
      ExtInfo.BranchAfters.push_back(std::make_pair(Block, Index));
  }

  /// Return the number of unique branch-afters on this scope.
  unsigned getNumBranchAfters() const {
    return ExtInfo ? ExtInfo->BranchAfters.size() : 0;
  }

  llvm::BasicBlock *getBranchAfterBlock(unsigned I) const {
    assert(I < getNumBranchAfters());
    return ExtInfo->BranchAfters[I].first;
  }

  llvm::ConstantInt *getBranchAfterIndex(unsigned I) const {
    assert(I < getNumBranchAfters());
    return ExtInfo->BranchAfters[I].second;
  }

  /// Add a branch-through to this cleanup scope.  A branch-through is
  /// a branch from a scope protected by this (normal) cleanup to an
  /// enclosing scope other than the immediately-enclosing normal
  /// cleanup scope.
  ///
  /// In the following example, the branch through B's scope is a
  /// branch-through, while the branch through A's scope is a
  /// branch-after:
  ///   for (;;) { A a; B b; break; }
  ///
  /// All branch-throughs have a common destination out of the
  /// cleanup, one possibly shared with the fall-through.  Therefore
  /// branch-throughs usually don't force a switch after the cleanup.
  ///
  /// \return true if the branch-through was new to this scope
  bool addBranchThrough(llvm::BasicBlock *Block) {
    return getExtInfo().Branches.insert(Block);
  }

  /// Determines if this cleanup scope has any branch throughs.
  bool hasBranchThroughs() const {
    if (!ExtInfo) return false;
    return (ExtInfo->BranchAfters.size() != ExtInfo->Branches.size());
  }

  static bool classof(const EHScope *Scope) {
    return (Scope->getKind() == Cleanup);
  }
};

/// An exceptions scope which filters exceptions thrown through it.
/// Only exceptions matching the filter types will be permitted to be
/// thrown.
///
/// This is used to implement C++ exception specifications.
class EHFilterScope : public EHScope {
  // Essentially ends in a flexible array member:
  // llvm::Value *FilterTypes[0];

  llvm::Value **getFilters() {
    return reinterpret_cast<llvm::Value**>(this+1);
  }

  llvm::Value * const *getFilters() const {
    return reinterpret_cast<llvm::Value* const *>(this+1);
  }

public:
  EHFilterScope(unsigned numFilters)
    : EHScope(Filter, EHScopeStack::stable_end()) {
    FilterBits.NumFilters = numFilters;
  }

  static size_t getSizeForNumFilters(unsigned numFilters) {
    return sizeof(EHFilterScope) + numFilters * sizeof(llvm::Value*);
  }

  unsigned getNumFilters() const { return FilterBits.NumFilters; }

  void setFilter(unsigned i, llvm::Value *filterValue) {
    assert(i < getNumFilters());
    getFilters()[i] = filterValue;
  }

  llvm::Value *getFilter(unsigned i) const {
    assert(i < getNumFilters());
    return getFilters()[i];
  }

  static bool classof(const EHScope *scope) {
    return scope->getKind() == Filter;
  }
};

/// An exceptions scope which calls std::terminate if any exception
/// reaches it.
class EHTerminateScope : public EHScope {
public:
  EHTerminateScope(EHScopeStack::stable_iterator enclosingEHScope)
    : EHScope(Terminate, enclosingEHScope) {}
  static size_t getSize() { return sizeof(EHTerminateScope); }

  static bool classof(const EHScope *scope) {
    return scope->getKind() == Terminate;
  }
};

/// A non-stable pointer into the scope stack.
class EHScopeStack::iterator {
  char *Ptr;

  friend class EHScopeStack;
  explicit iterator(char *Ptr) : Ptr(Ptr) {}

public:
  iterator() : Ptr(0) {}

  EHScope *get() const { 
    return reinterpret_cast<EHScope*>(Ptr);
  }

  EHScope *operator->() const { return get(); }
  EHScope &operator*() const { return *get(); }

  iterator &operator++() {
    switch (get()->getKind()) {
    case EHScope::Catch:
      Ptr += EHCatchScope::getSizeForNumHandlers(
          static_cast<const EHCatchScope*>(get())->getNumHandlers());
      break;

    case EHScope::Filter:
      Ptr += EHFilterScope::getSizeForNumFilters(
          static_cast<const EHFilterScope*>(get())->getNumFilters());
      break;

    case EHScope::Cleanup:
      Ptr += static_cast<const EHCleanupScope*>(get())
        ->getAllocatedSize();
      break;

    case EHScope::Terminate:
      Ptr += EHTerminateScope::getSize();
      break;
    }

    return *this;
  }

  iterator next() {
    iterator copy = *this;
    ++copy;
    return copy;
  }

  iterator operator++(int) {
    iterator copy = *this;
    operator++();
    return copy;
  }

  bool encloses(iterator other) const { return Ptr >= other.Ptr; }
  bool strictlyEncloses(iterator other) const { return Ptr > other.Ptr; }

  bool operator==(iterator other) const { return Ptr == other.Ptr; }
  bool operator!=(iterator other) const { return Ptr != other.Ptr; }
};

inline EHScopeStack::iterator EHScopeStack::begin() const {
  return iterator(StartOfData);
}

inline EHScopeStack::iterator EHScopeStack::end() const {
  return iterator(EndOfBuffer);
}

inline void EHScopeStack::popCatch() {
  assert(!empty() && "popping exception stack when not empty");

  EHCatchScope &scope = cast<EHCatchScope>(*begin());
  InnermostEHScope = scope.getEnclosingEHScope();
  StartOfData += EHCatchScope::getSizeForNumHandlers(scope.getNumHandlers());
}

inline void EHScopeStack::popTerminate() {
  assert(!empty() && "popping exception stack when not empty");

  EHTerminateScope &scope = cast<EHTerminateScope>(*begin());
  InnermostEHScope = scope.getEnclosingEHScope();
  StartOfData += EHTerminateScope::getSize();
}

inline EHScopeStack::iterator EHScopeStack::find(stable_iterator sp) const {
  assert(sp.isValid() && "finding invalid savepoint");
  assert(sp.Size <= stable_begin().Size && "finding savepoint after pop");
  return iterator(EndOfBuffer - sp.Size);
}

inline EHScopeStack::stable_iterator
EHScopeStack::stabilize(iterator ir) const {
  assert(StartOfData <= ir.Ptr && ir.Ptr <= EndOfBuffer);
  return stable_iterator(EndOfBuffer - ir.Ptr);
}

}
}

#endif
