//===-- CIRGenCleanup.h - Classes for cleanups CIR generation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes support the generation of CIR for cleanups, initially based
// on LLVM IR cleanup handling, but ought to change as CIR evolves.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CGCLEANUP_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CGCLEANUP_H

#include "Address.h"
#include "EHScopeStack.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
class FunctionDecl;
}

namespace cir {
class CIRGenModule;
class CIRGenFunction;

/// The MS C++ ABI needs a pointer to RTTI data plus some flags to describe the
/// type of a catch handler, so we use this wrapper.
struct CatchTypeInfo {
  mlir::TypedAttr RTTI;
  unsigned Flags;
};

/// A protected scope for zero-cost EH handling.
class EHScope {
  mlir::Block *CachedLandingPad;
  mlir::Block *CachedEHDispatchBlock;

  EHScopeStack::stable_iterator EnclosingEHScope;

  class CommonBitFields {
    friend class EHScope;
    unsigned Kind : 3;
  };
  enum { NumCommonBits = 3 };

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

    /// Whether this cleanup is a lifetime marker
    unsigned IsLifetimeMarker : 1;

    /// Whether the normal cleanup should test the activation flag.
    unsigned TestFlagInNormalCleanup : 1;

    /// Whether the EH cleanup should test the activation flag.
    unsigned TestFlagInEHCleanup : 1;

    /// The amount of extra storage needed by the Cleanup.
    /// Always a multiple of the scope-stack alignment.
    unsigned CleanupSize : 12;
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
      : CachedLandingPad(nullptr), CachedEHDispatchBlock(nullptr),
        EnclosingEHScope(enclosingEHScope) {
    CommonBits.Kind = kind;
  }

  Kind getKind() const { return static_cast<Kind>(CommonBits.Kind); }

  mlir::Block *getCachedLandingPad() const { return CachedLandingPad; }

  void setCachedLandingPad(mlir::Block *block) { CachedLandingPad = block; }

  mlir::Block *getCachedEHDispatchBlock() const {
    return CachedEHDispatchBlock;
  }

  void setCachedEHDispatchBlock(mlir::Block *block) {
    CachedEHDispatchBlock = block;
  }

  bool hasEHBranches() const {
    if (mlir::Block *block = getCachedEHDispatchBlock())
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
    CatchTypeInfo Type;

    /// The catch handler for this type.
    mlir::Block *Block;

    bool isCatchAll() const { return Type.RTTI == nullptr; }
  };

private:
  friend class EHScopeStack;

  Handler *getHandlers() { return reinterpret_cast<Handler *>(this + 1); }

  const Handler *getHandlers() const {
    return reinterpret_cast<const Handler *>(this + 1);
  }

public:
  static size_t getSizeForNumHandlers(unsigned N) {
    return sizeof(EHCatchScope) + N * sizeof(Handler);
  }

  EHCatchScope(unsigned numHandlers,
               EHScopeStack::stable_iterator enclosingEHScope)
      : EHScope(Catch, enclosingEHScope) {
    CatchBits.NumHandlers = numHandlers;
    assert(CatchBits.NumHandlers == numHandlers && "NumHandlers overflow?");
  }

  unsigned getNumHandlers() const { return CatchBits.NumHandlers; }

  void setCatchAllHandler(unsigned I, mlir::Block *Block) {
    setHandler(I, CatchTypeInfo{nullptr, 0}, Block);
  }

  void setHandler(unsigned I, mlir::TypedAttr Type, mlir::Block *Block) {
    assert(I < getNumHandlers());
    getHandlers()[I].Type = CatchTypeInfo{Type, 0};
    getHandlers()[I].Block = Block;
  }

  void setHandler(unsigned I, CatchTypeInfo Type, mlir::Block *Block) {
    assert(I < getNumHandlers());
    getHandlers()[I].Type = Type;
    getHandlers()[I].Block = Block;
  }

  const Handler &getHandler(unsigned I) const {
    assert(I < getNumHandlers());
    return getHandlers()[I];
  }

  // Clear all handler blocks.
  // FIXME: it's better to always call clearHandlerBlocks in DTOR and have a
  // 'takeHandler' or some such function which removes ownership from the
  // EHCatchScope object if the handlers should live longer than EHCatchScope.
  void clearHandlerBlocks() {
    for (unsigned I = 0, N = getNumHandlers(); I != N; ++I)
      delete getHandler(I).Block;
  }

  typedef const Handler *iterator;
  iterator begin() const { return getHandlers(); }
  iterator end() const { return getHandlers() + getNumHandlers(); }

  static bool classof(const EHScope *Scope) {
    return Scope->getKind() == Catch;
  }
};

/// A cleanup scope which generates the cleanup blocks lazily.
class alignas(8) EHCleanupScope : public EHScope {
  /// The nearest normal cleanup scope enclosing this one.
  EHScopeStack::stable_iterator EnclosingNormal;

  /// The nearest EH scope enclosing this one.
  EHScopeStack::stable_iterator EnclosingEH;

  /// The dual entry/exit block along the normal edge.  This is lazily
  /// created if needed before the cleanup is popped.
  mlir::Block *NormalBlock;

  /// An optional i1 variable indicating whether this cleanup has been
  /// activated yet.
  Address ActiveFlag;

  /// Extra information required for cleanups that have resolved
  /// branches through them.  This has to be allocated on the side
  /// because everything on the cleanup stack has be trivially
  /// movable.
  struct ExtInfo {
    /// The destinations of normal branch-afters and branch-throughs.
    llvm::SmallPtrSet<mlir::Block *, 4> Branches;

    /// Normal branch-afters.
    llvm::SmallVector<std::pair<mlir::Block *, mlir::Value>, 4> BranchAfters;
  };
  mutable struct ExtInfo *ExtInfo;

  /// The number of fixups required by enclosing scopes (not including
  /// this one).  If this is the top cleanup scope, all the fixups
  /// from this index onwards belong to this scope.
  unsigned FixupDepth;

  struct ExtInfo &getExtInfo() {
    if (!ExtInfo)
      ExtInfo = new struct ExtInfo();
    return *ExtInfo;
  }

  const struct ExtInfo &getExtInfo() const {
    if (!ExtInfo)
      ExtInfo = new struct ExtInfo();
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

  EHCleanupScope(bool isNormal, bool isEH, unsigned cleanupSize,
                 unsigned fixupDepth,
                 EHScopeStack::stable_iterator enclosingNormal,
                 EHScopeStack::stable_iterator enclosingEH)
      : EHScope(EHScope::Cleanup, enclosingEH),
        EnclosingNormal(enclosingNormal), NormalBlock(nullptr),
        ActiveFlag(Address::invalid()), ExtInfo(nullptr),
        FixupDepth(fixupDepth) {
    CleanupBits.IsNormalCleanup = isNormal;
    CleanupBits.IsEHCleanup = isEH;
    CleanupBits.IsActive = true;
    CleanupBits.IsLifetimeMarker = false;
    CleanupBits.TestFlagInNormalCleanup = false;
    CleanupBits.TestFlagInEHCleanup = false;
    CleanupBits.CleanupSize = cleanupSize;

    assert(CleanupBits.CleanupSize == cleanupSize && "cleanup size overflow");
  }

  void Destroy() { delete ExtInfo; }
  // Objects of EHCleanupScope are not destructed. Use Destroy().
  ~EHCleanupScope() = delete;

  bool isNormalCleanup() const { return CleanupBits.IsNormalCleanup; }
  mlir::Block *getNormalBlock() const { return NormalBlock; }
  void setNormalBlock(mlir::Block *BB) { NormalBlock = BB; }

  bool isEHCleanup() const { return CleanupBits.IsEHCleanup; }

  bool isActive() const { return CleanupBits.IsActive; }
  void setActive(bool A) { CleanupBits.IsActive = A; }

  bool isLifetimeMarker() const { return CleanupBits.IsLifetimeMarker; }
  void setLifetimeMarker() { CleanupBits.IsLifetimeMarker = true; }

  bool hasActiveFlag() const { return ActiveFlag.isValid(); }
  Address getActiveFlag() const { return ActiveFlag; }
  void setActiveFlag(Address Var) {
    assert(Var.getAlignment().isOne());
    ActiveFlag = Var;
  }

  void setTestFlagInNormalCleanup() {
    CleanupBits.TestFlagInNormalCleanup = true;
  }
  bool shouldTestFlagInNormalCleanup() const {
    return CleanupBits.TestFlagInNormalCleanup;
  }

  void setTestFlagInEHCleanup() { CleanupBits.TestFlagInEHCleanup = true; }
  bool shouldTestFlagInEHCleanup() const {
    return CleanupBits.TestFlagInEHCleanup;
  }

  unsigned getFixupDepth() const { return FixupDepth; }
  EHScopeStack::stable_iterator getEnclosingNormalCleanup() const {
    return EnclosingNormal;
  }

  size_t getCleanupSize() const { return CleanupBits.CleanupSize; }
  void *getCleanupBuffer() { return this + 1; }

  EHScopeStack::Cleanup *getCleanup() {
    return reinterpret_cast<EHScopeStack::Cleanup *>(getCleanupBuffer());
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
  void addBranchAfter(mlir::Value Index, mlir::Block *Block) {
    struct ExtInfo &ExtInfo = getExtInfo();
    if (ExtInfo.Branches.insert(Block).second)
      ExtInfo.BranchAfters.push_back(std::make_pair(Block, Index));
  }

  /// Return the number of unique branch-afters on this scope.
  unsigned getNumBranchAfters() const {
    return ExtInfo ? ExtInfo->BranchAfters.size() : 0;
  }

  mlir::Block *getBranchAfterBlock(unsigned I) const {
    assert(I < getNumBranchAfters());
    return ExtInfo->BranchAfters[I].first;
  }

  mlir::Value getBranchAfterIndex(unsigned I) const {
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
  bool addBranchThrough(mlir::Block *Block) {
    return getExtInfo().Branches.insert(Block).second;
  }

  /// Determines if this cleanup scope has any branch throughs.
  bool hasBranchThroughs() const {
    if (!ExtInfo)
      return false;
    return (ExtInfo->BranchAfters.size() != ExtInfo->Branches.size());
  }

  static bool classof(const EHScope *Scope) {
    return (Scope->getKind() == Cleanup);
  }
};
// NOTE: there's a bunch of different data classes tacked on after an
// EHCleanupScope. It is asserted (in EHScopeStack::pushCleanup*) that
// they don't require greater alignment than ScopeStackAlignment. So,
// EHCleanupScope ought to have alignment equal to that -- not more
// (would be misaligned by the stack allocator), and not less (would
// break the appended classes).
static_assert(alignof(EHCleanupScope) == EHScopeStack::ScopeStackAlignment,
              "EHCleanupScope expected alignment");

/// An exceptions scope which filters exceptions thrown through it.
/// Only exceptions matching the filter types will be permitted to be
/// thrown.
///
/// This is used to implement C++ exception specifications.
class EHFilterScope : public EHScope {
  // Essentially ends in a flexible array member:
  // mlir::Value FilterTypes[0];

  mlir::Value *getFilters() {
    return reinterpret_cast<mlir::Value *>(this + 1);
  }

  mlir::Value const *getFilters() const {
    return reinterpret_cast<mlir::Value const *>(this + 1);
  }

public:
  EHFilterScope(unsigned numFilters)
      : EHScope(Filter, EHScopeStack::stable_end()) {
    FilterBits.NumFilters = numFilters;
    assert(FilterBits.NumFilters == numFilters && "NumFilters overflow");
  }

  static size_t getSizeForNumFilters(unsigned numFilters) {
    return sizeof(EHFilterScope) + numFilters * sizeof(mlir::Value);
  }

  unsigned getNumFilters() const { return FilterBits.NumFilters; }

  void setFilter(unsigned i, mlir::Value filterValue) {
    assert(i < getNumFilters());
    getFilters()[i] = filterValue;
  }

  mlir::Value getFilter(unsigned i) const {
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
  iterator() : Ptr(nullptr) {}

  EHScope *get() const { return reinterpret_cast<EHScope *>(Ptr); }

  EHScope *operator->() const { return get(); }
  EHScope &operator*() const { return *get(); }

  iterator &operator++() {
    size_t Size;
    switch (get()->getKind()) {
    case EHScope::Catch:
      Size = EHCatchScope::getSizeForNumHandlers(
          static_cast<const EHCatchScope *>(get())->getNumHandlers());
      break;

    case EHScope::Filter:
      Size = EHFilterScope::getSizeForNumFilters(
          static_cast<const EHFilterScope *>(get())->getNumFilters());
      break;

    case EHScope::Cleanup:
      Size = static_cast<const EHCleanupScope *>(get())->getAllocatedSize();
      break;

    case EHScope::Terminate:
      Size = EHTerminateScope::getSize();
      break;
    }
    Ptr += llvm::alignTo(Size, ScopeStackAlignment);
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

  EHCatchScope &scope = llvm::cast<EHCatchScope>(*begin());
  InnermostEHScope = scope.getEnclosingEHScope();
  deallocate(EHCatchScope::getSizeForNumHandlers(scope.getNumHandlers()));
}

inline void EHScopeStack::popTerminate() {
  assert(!empty() && "popping exception stack when not empty");

  EHTerminateScope &scope = llvm::cast<EHTerminateScope>(*begin());
  InnermostEHScope = scope.getEnclosingEHScope();
  deallocate(EHTerminateScope::getSize());
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

/// The exceptions personality for a function.
struct EHPersonality {
  const char *PersonalityFn;

  // If this is non-null, this personality requires a non-standard
  // function for rethrowing an exception after a catchall cleanup.
  // This function must have prototype void(void*).
  const char *CatchallRethrowFn;

  static const EHPersonality &get(CIRGenModule &CGM,
                                  const clang::FunctionDecl *FD);
  static const EHPersonality &get(CIRGenFunction &CGF);

  static const EHPersonality GNU_C;
  static const EHPersonality GNU_C_SJLJ;
  static const EHPersonality GNU_C_SEH;
  static const EHPersonality GNU_ObjC;
  static const EHPersonality GNU_ObjC_SJLJ;
  static const EHPersonality GNU_ObjC_SEH;
  static const EHPersonality GNUstep_ObjC;
  static const EHPersonality GNU_ObjCXX;
  static const EHPersonality NeXT_ObjC;
  static const EHPersonality GNU_CPlusPlus;
  static const EHPersonality GNU_CPlusPlus_SJLJ;
  static const EHPersonality GNU_CPlusPlus_SEH;
  static const EHPersonality MSVC_except_handler;
  static const EHPersonality MSVC_C_specific_handler;
  static const EHPersonality MSVC_CxxFrameHandler3;
  static const EHPersonality GNU_Wasm_CPlusPlus;
  static const EHPersonality XL_CPlusPlus;

  /// Does this personality use landingpads or the family of pad instructions
  /// designed to form funclets?
  bool usesFuncletPads() const {
    return isMSVCPersonality() || isWasmPersonality();
  }

  bool isMSVCPersonality() const {
    return this == &MSVC_except_handler || this == &MSVC_C_specific_handler ||
           this == &MSVC_CxxFrameHandler3;
  }

  bool isWasmPersonality() const { return this == &GNU_Wasm_CPlusPlus; }

  bool isMSVCXXPersonality() const { return this == &MSVC_CxxFrameHandler3; }
};
} // namespace cir

#endif
