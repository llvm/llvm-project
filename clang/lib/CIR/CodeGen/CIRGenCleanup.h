//===----------------------------------------------------------------------===//
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

#ifndef CLANG_LIB_CIR_CODEGEN_CIRGENCLEANUP_H
#define CLANG_LIB_CIR_CODEGEN_CIRGENCLEANUP_H

#include "Address.h"
#include "EHScopeStack.h"
#include "mlir/IR/Value.h"

namespace clang::CIRGen {

/// A protected scope for zero-cost EH handling.
class EHScope {
  class CommonBitFields {
    friend class EHScope;
    unsigned kind : 3;
  };
  enum { NumCommonBits = 3 };

protected:
  class CleanupBitFields {
    friend class EHCleanupScope;
    unsigned : NumCommonBits;

    /// Whether this cleanup needs to be run along normal edges.
    unsigned isNormalCleanup : 1;

    /// Whether this cleanup needs to be run along exception edges.
    unsigned isEHCleanup : 1;

    /// Whether this cleanup is currently active.
    unsigned isActive : 1;

    /// Whether this cleanup is a lifetime marker
    unsigned isLifetimeMarker : 1;

    /// Whether the normal cleanup should test the activation flag.
    unsigned testFlagInNormalCleanup : 1;

    /// Whether the EH cleanup should test the activation flag.
    unsigned testFlagInEHCleanup : 1;

    /// The amount of extra storage needed by the Cleanup.
    /// Always a multiple of the scope-stack alignment.
    unsigned cleanupSize : 12;
  };

  union {
    CommonBitFields commonBits;
    CleanupBitFields cleanupBits;
  };

public:
  enum Kind { Cleanup, Catch, Terminate, Filter };

  EHScope(Kind kind) { commonBits.kind = kind; }

  Kind getKind() const { return static_cast<Kind>(commonBits.kind); }
};

/// A cleanup scope which generates the cleanup blocks lazily.
class alignas(EHScopeStack::ScopeStackAlignment) EHCleanupScope
    : public EHScope {
  /// The nearest normal cleanup scope enclosing this one.
  EHScopeStack::stable_iterator enclosingNormal;

  /// The dual entry/exit block along the normal edge.  This is lazily
  /// created if needed before the cleanup is popped.
  mlir::Block *normalBlock = nullptr;

  /// The number of fixups required by enclosing scopes (not including
  /// this one).  If this is the top cleanup scope, all the fixups
  /// from this index onwards belong to this scope.
  unsigned fixupDepth = 0;

public:
  /// Gets the size required for a lazy cleanup scope with the given
  /// cleanup-data requirements.
  static size_t getSizeForCleanupSize(size_t size) {
    return sizeof(EHCleanupScope) + size;
  }

  size_t getAllocatedSize() const {
    return sizeof(EHCleanupScope) + cleanupBits.cleanupSize;
  }

  EHCleanupScope(unsigned cleanupSize, unsigned fixupDepth,
                 EHScopeStack::stable_iterator enclosingNormal)
      : EHScope(EHScope::Cleanup), enclosingNormal(enclosingNormal),
        fixupDepth(fixupDepth) {
    // TODO(cir): When exception handling is upstreamed, isNormalCleanup and
    // isEHCleanup will be arguments to the constructor.
    cleanupBits.isNormalCleanup = true;
    cleanupBits.isEHCleanup = false;
    cleanupBits.isActive = true;
    cleanupBits.isLifetimeMarker = false;
    cleanupBits.testFlagInNormalCleanup = false;
    cleanupBits.testFlagInEHCleanup = false;
    cleanupBits.cleanupSize = cleanupSize;

    assert(cleanupBits.cleanupSize == cleanupSize && "cleanup size overflow");
  }

  void destroy() {}
  // Objects of EHCleanupScope are not destructed. Use destroy().
  ~EHCleanupScope() = delete;

  mlir::Block *getNormalBlock() const { return normalBlock; }
  void setNormalBlock(mlir::Block *bb) { normalBlock = bb; }

  bool isNormalCleanup() const { return cleanupBits.isNormalCleanup; }

  bool isActive() const { return cleanupBits.isActive; }
  void setActive(bool isActive) { cleanupBits.isActive = isActive; }

  unsigned getFixupDepth() const { return fixupDepth; }
  EHScopeStack::stable_iterator getEnclosingNormalCleanup() const {
    return enclosingNormal;
  }

  size_t getCleanupSize() const { return cleanupBits.cleanupSize; }
  void *getCleanupBuffer() { return this + 1; }

  EHScopeStack::Cleanup *getCleanup() {
    return reinterpret_cast<EHScopeStack::Cleanup *>(getCleanupBuffer());
  }

  static bool classof(const EHScope *scope) {
    return (scope->getKind() == Cleanup);
  }

  void markEmitted() {}
};

/// A non-stable pointer into the scope stack.
class EHScopeStack::iterator {
  char *ptr = nullptr;

  friend class EHScopeStack;
  explicit iterator(char *ptr) : ptr(ptr) {}

public:
  iterator() = default;

  EHScope *get() const { return reinterpret_cast<EHScope *>(ptr); }

  EHScope &operator*() const { return *get(); }
};

inline EHScopeStack::iterator EHScopeStack::begin() const {
  return iterator(startOfData);
}

inline EHScopeStack::iterator
EHScopeStack::find(stable_iterator savePoint) const {
  assert(savePoint.isValid() && "finding invalid savepoint");
  assert(savePoint.size <= stable_begin().size &&
         "finding savepoint after pop");
  return iterator(endOfBuffer - savePoint.size);
}

} // namespace clang::CIRGen
#endif // CLANG_LIB_CIR_CODEGEN_CIRGENCLEANUP_H
