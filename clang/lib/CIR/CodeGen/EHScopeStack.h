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
public:
  /// Information for lazily generating a cleanup.  Subclasses must be
  /// POD-like: cleanups will not be destructed, and they will be
  /// allocated on the cleanup stack and freely copied and moved
  /// around.
  ///
  /// Cleanup implementations should generally be declared in an
  /// anonymous namespace.
  class Cleanup {
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

  // Classic codegen has a finely tuned custom allocator and a complex stack
  // management scheme. We'll probably eventually want to find a way to share
  // that implementation. For now, we will use a very simplified implementation
  // to get cleanups working.
  llvm::SmallVector<std::unique_ptr<Cleanup>, 8> cleanupStack;

private:
  /// The CGF this Stack belong to
  CIRGenFunction *cgf = nullptr;

public:
  EHScopeStack() = default;
  ~EHScopeStack() = default;

  /// Push a lazily-created cleanup on the stack.
  template <class T, class... As> void pushCleanup(CleanupKind kind, As... a) {
    cleanupStack.push_back(std::make_unique<T>(a...));
  }

  void setCGF(CIRGenFunction *inCGF) { cgf = inCGF; }

  size_t getStackDepth() const { return cleanupStack.size(); }
};

} // namespace clang::CIRGen

#endif // CLANG_LIB_CIR_CODEGEN_EHSCOPESTACK_H
