//===-- EHScopeStack.h - Stack for cleanup CIR generation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes should be the minimum interface required for other parts of
// CIR CodeGen to emit cleanups.  The implementation is in CIRGenCleanup.cpp and
// other implemenentation details that are not widely needed are in
// CIRGenCleanup.h.
//
// TODO(cir): this header should be shared between LLVM and CIR codegen.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_CODEGEN_EHSCOPESTACK_H
#define CLANG_LIB_CIR_CODEGEN_EHSCOPESTACK_H

#include "llvm/ADT/SmallVector.h"

namespace clang::CIRGen {

class CIRGenFunction;

enum CleanupKind : unsigned {
  /// Denotes a cleanup that should run when a scope is exited using exceptional
  /// control flow (a throw statement leading to stack unwinding, ).
  EHCleanup = 0x1,

  /// Denotes a cleanup that should run when a scope is exited using normal
  /// control flow (falling off the end of the scope, return, goto, ...).
  NormalCleanup = 0x2,

  NormalAndEHCleanup = EHCleanup | NormalCleanup,

  LifetimeMarker = 0x8,
  NormalEHLifetimeMarker = LifetimeMarker | NormalAndEHCleanup,
};

/// A stack of scopes which respond to exceptions, including cleanups
/// and catch blocks.
class EHScopeStack {
  friend class CIRGenFunction;

public:
  // TODO(ogcg): Switch to alignof(uint64_t) instead of 8
  enum { ScopeStackAlignment = 8 };

  /// A saved depth on the scope stack.  This is necessary because
  /// pushing scopes onto the stack invalidates iterators.
  class stable_iterator {
    friend class EHScopeStack;

    /// Offset from startOfData to endOfBuffer.
    ptrdiff_t size = -1;

    explicit stable_iterator(ptrdiff_t size) : size(size) {}

  public:
    static stable_iterator invalid() { return stable_iterator(-1); }
    stable_iterator() = default;

    bool isValid() const { return size >= 0; }

    /// Returns true if this scope encloses I.
    /// Returns false if I is invalid.
    /// This scope must be valid.
    bool encloses(stable_iterator other) const { return size <= other.size; }

    /// Returns true if this scope strictly encloses I: that is,
    /// if it encloses I and is not I.
    /// Returns false is I is invalid.
    /// This scope must be valid.
    bool strictlyEncloses(stable_iterator I) const { return size < I.size; }

    friend bool operator==(stable_iterator A, stable_iterator B) {
      return A.size == B.size;
    }
    friend bool operator!=(stable_iterator A, stable_iterator B) {
      return A.size != B.size;
    }
  };

  /// Information for lazily generating a cleanup.  Subclasses must be
  /// POD-like: cleanups will not be destructed, and they will be
  /// allocated on the cleanup stack and freely copied and moved
  /// around.
  ///
  /// Cleanup implementations should generally be declared in an
  /// anonymous namespace.
  class LLVM_MOVABLE_POLYMORPHIC_TYPE Cleanup {
    // Anchor the construction vtable.
    virtual void anchor();

  public:
    Cleanup(const Cleanup &) = default;
    Cleanup(Cleanup &&) {}
    Cleanup() = default;

    virtual ~Cleanup() = default;

    /// Emit the cleanup.  For normal cleanups, this is run in the
    /// same EH context as when the cleanup was pushed, i.e. the
    /// immediately-enclosing context of the cleanup scope.  For
    /// EH cleanups, this is run in a terminate context.
    ///
    // \param flags cleanup kind.
    virtual void emit(CIRGenFunction &cgf) = 0;
  };

private:
  // The implementation for this class is in CIRGenCleanup.h and
  // CIRGenCleanup.cpp; the definition is here because it's used as a
  // member of CIRGenFunction.

  /// The start of the scope-stack buffer, i.e. the allocated pointer
  /// for the buffer.  All of these pointers are either simultaneously
  /// null or simultaneously valid.
  std::unique_ptr<char[]> startOfBuffer;

  /// The end of the buffer.
  char *endOfBuffer = nullptr;

  /// The first valid entry in the buffer.
  char *startOfData = nullptr;

  /// The CGF this Stack belong to
  CIRGenFunction *cgf = nullptr;

  // This class uses a custom allocator for maximum efficiency because cleanups
  // are allocated and freed very frequently. It's basically a bump pointer
  // allocator, but we can't use LLVM's BumpPtrAllocator because we use offsets
  // into the buffer as stable iterators.
  char *allocate(size_t size);
  void deallocate(size_t size);

  void *pushCleanup(CleanupKind kind, size_t dataSize);

public:
  EHScopeStack() = default;
  ~EHScopeStack() = default;

  /// Push a lazily-created cleanup on the stack.
  template <class T, class... As> void pushCleanup(CleanupKind kind, As... a) {
    static_assert(alignof(T) <= ScopeStackAlignment,
                  "Cleanup's alignment is too large.");
    void *buffer = pushCleanup(kind, sizeof(T));
    [[maybe_unused]] Cleanup *obj = new (buffer) T(a...);
  }

  void setCGF(CIRGenFunction *inCGF) { cgf = inCGF; }

  /// Pops a cleanup scope off the stack.  This is private to CIRGenCleanup.cpp.
  void popCleanup();

  /// Determines whether the exception-scopes stack is empty.
  bool empty() const { return startOfData == endOfBuffer; }

  /// An unstable reference to a scope-stack depth.  Invalidated by
  /// pushes but not pops.
  class iterator;

  /// Returns an iterator pointing to the innermost EH scope.
  iterator begin() const;

  /// Create a stable reference to the top of the EH stack.  The
  /// returned reference is valid until that scope is popped off the
  /// stack.
  stable_iterator stable_begin() const {
    return stable_iterator(endOfBuffer - startOfData);
  }

  /// Turn a stable reference to a scope depth into a unstable pointer
  /// to the EH stack.
  iterator find(stable_iterator savePoint) const;

  /// Create a stable reference to the bottom of the EH stack.
  static stable_iterator stable_end() { return stable_iterator(0); }
};

} // namespace clang::CIRGen

#endif // CLANG_LIB_CIR_CODEGEN_EHSCOPESTACK_H
