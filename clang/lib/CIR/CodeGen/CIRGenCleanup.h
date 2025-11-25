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
#include "CIRGenModule.h"
#include "EHScopeStack.h"
#include "mlir/IR/Value.h"
#include "clang/AST/StmtCXX.h"

namespace clang::CIRGen {

/// The MS C++ ABI needs a pointer to RTTI data plus some flags to describe the
/// type of a catch handler, so we use this wrapper.
struct CatchTypeInfo {
  mlir::TypedAttr rtti;
  unsigned flags;
};

/// A protected scope for zero-cost EH handling.
class EHScope {
  EHScopeStack::stable_iterator enclosingEHScope;

  class CommonBitFields {
    friend class EHScope;
    unsigned kind : 3;
  };
  enum { NumCommonBits = 3 };

  bool scopeMayThrow;

protected:
  class CatchBitFields {
    friend class EHCatchScope;
    unsigned : NumCommonBits;
    unsigned numHandlers : 32 - NumCommonBits;
  };

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
    CatchBitFields catchBits;
    CleanupBitFields cleanupBits;
  };

public:
  enum Kind { Cleanup, Catch, Terminate, Filter };

  EHScope(Kind kind, EHScopeStack::stable_iterator enclosingEHScope)
      : enclosingEHScope(enclosingEHScope) {
    commonBits.kind = kind;
  }

  Kind getKind() const { return static_cast<Kind>(commonBits.kind); }

  bool mayThrow() const {
    // Traditional LLVM codegen also checks for `!block->use_empty()`, but
    // in CIRGen the block content is not important, just used as a way to
    // signal `hasEHBranches`.
    return scopeMayThrow;
  }

  void setMayThrow(bool mayThrow) { scopeMayThrow = mayThrow; }

  EHScopeStack::stable_iterator getEnclosingEHScope() const {
    return enclosingEHScope;
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
    /// A type info value, or null MLIR attribute for a catch-all
    CatchTypeInfo type;

    /// The catch handler for this type.
    mlir::Region *region;

    /// The catch handler stmt.
    const CXXCatchStmt *stmt;

    bool isCatchAll() const { return type.rtti == nullptr; }
  };

private:
  friend class EHScopeStack;

  Handler *getHandlers() { return reinterpret_cast<Handler *>(this + 1); }

  const Handler *getHandlers() const {
    return reinterpret_cast<const Handler *>(this + 1);
  }

public:
  static size_t getSizeForNumHandlers(unsigned n) {
    return sizeof(EHCatchScope) + n * sizeof(Handler);
  }

  EHCatchScope(unsigned numHandlers,
               EHScopeStack::stable_iterator enclosingEHScope)
      : EHScope(Catch, enclosingEHScope) {
    catchBits.numHandlers = numHandlers;
    assert(catchBits.numHandlers == numHandlers && "NumHandlers overflow?");
  }

  unsigned getNumHandlers() const { return catchBits.numHandlers; }

  void setHandler(unsigned i, CatchTypeInfo type, mlir::Region *region,
                  const CXXCatchStmt *stmt) {
    assert(i < getNumHandlers());
    Handler *handler = &getHandlers()[i];
    handler->type = type;
    handler->region = region;
    handler->stmt = stmt;
  }

  const Handler &getHandler(unsigned i) const {
    assert(i < getNumHandlers());
    return getHandlers()[i];
  }

  // Clear all handler blocks.
  // FIXME: it's better to always call clearHandlerBlocks in DTOR and have a
  // 'takeHandler' or some such function which removes ownership from the
  // EHCatchScope object if the handlers should live longer than EHCatchScope.
  void clearHandlerBlocks() {
    // The blocks are owned by TryOp, nothing to delete.
  }

  using iterator = const Handler *;
  iterator begin() const { return getHandlers(); }
  iterator end() const { return getHandlers() + getNumHandlers(); }

  static bool classof(const EHScope *scope) {
    return scope->getKind() == Catch;
  }
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
                 EHScopeStack::stable_iterator enclosingNormal,
                 EHScopeStack::stable_iterator enclosingEH)
      : EHScope(EHScope::Cleanup, enclosingEH),
        enclosingNormal(enclosingNormal), fixupDepth(fixupDepth) {
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

  bool isLifetimeMarker() const { return cleanupBits.isLifetimeMarker; }

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

  EHScope *operator->() const { return get(); }
  EHScope &operator*() const { return *get(); }

  iterator &operator++() {
    size_t size;
    switch (get()->getKind()) {
    case EHScope::Catch:
      size = EHCatchScope::getSizeForNumHandlers(
          static_cast<const EHCatchScope *>(get())->getNumHandlers());
      break;

    case EHScope::Filter:
      llvm_unreachable("EHScopeStack::iterator Filter");
      break;

    case EHScope::Cleanup:
      llvm_unreachable("EHScopeStack::iterator Cleanup");
      break;

    case EHScope::Terminate:
      llvm_unreachable("EHScopeStack::iterator Terminate");
      break;
    }
    ptr += llvm::alignTo(size, ScopeStackAlignment);
    return *this;
  }

  bool operator==(iterator other) const { return ptr == other.ptr; }
  bool operator!=(iterator other) const { return ptr != other.ptr; }
};

inline EHScopeStack::iterator EHScopeStack::begin() const {
  return iterator(startOfData);
}

inline EHScopeStack::iterator EHScopeStack::end() const {
  return iterator(endOfBuffer);
}

inline EHScopeStack::iterator
EHScopeStack::find(stable_iterator savePoint) const {
  assert(savePoint.isValid() && "finding invalid savepoint");
  assert(savePoint.size <= stable_begin().size &&
         "finding savepoint after pop");
  return iterator(endOfBuffer - savePoint.size);
}

inline void EHScopeStack::popCatch() {
  assert(!empty() && "popping exception stack when not empty");

  EHCatchScope &scope = llvm::cast<EHCatchScope>(*begin());
  innermostEHScope = scope.getEnclosingEHScope();
  deallocate(EHCatchScope::getSizeForNumHandlers(scope.getNumHandlers()));
}

/// The exceptions personality for a function.
struct EHPersonality {
  const char *personalityFn = nullptr;

  // If this is non-null, this personality requires a non-standard
  // function for rethrowing an exception after a catchall cleanup.
  // This function must have prototype void(void*).
  const char *catchallRethrowFn = nullptr;

  static const EHPersonality &get(CIRGenModule &cgm,
                                  const clang::FunctionDecl *fd);
  static const EHPersonality &get(CIRGenFunction &cgf);

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
  static const EHPersonality ZOS_CPlusPlus;

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

} // namespace clang::CIRGen
#endif // CLANG_LIB_CIR_CODEGEN_CIRGENCLEANUP_H
