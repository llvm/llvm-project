//===-- StatementContext.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_STATEMENTCONTEXT_H
#define FORTRAN_LOWER_STATEMENTCONTEXT_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

namespace Fortran::lower {

/// When lowering a statement, temporaries for intermediate results may be
/// allocated on the heap.  A StatementContext enables their deallocation
/// either explicitly with finalize() calls, or implicitly at the end of
/// the context.  A context may prohibit temporary allocation.  Otherwise,
/// an initial "outer" context scope may have nested context scopes, which
/// must make explicit subscope finalize() calls.
class StatementContext {
public:
  explicit StatementContext(bool cleanupProhibited = false) {
    if (cleanupProhibited)
      return;
    cufs.push_back({});
  }

  ~StatementContext() {
    if (!cufs.empty())
      finalizeAndPop();
    assert(cufs.empty() && "invalid StatementContext destructor call");
  }

  using CleanupFunction = std::function<void()>;

  /// Push a context subscope.
  void pushScope() {
    assert(!cufs.empty() && "invalid pushScope statement context");
    cufs.push_back({});
  }

  /// Append a cleanup function to the "list" of cleanup functions.
  void attachCleanup(CleanupFunction cuf) {
    assert(!cufs.empty() && "invalid attachCleanup statement context");
    if (cufs.back()) {
      CleanupFunction oldCleanup = *cufs.back();
      cufs.back() = [=]() {
        cuf();
        oldCleanup();
      };
    } else {
      cufs.back() = cuf;
    }
  }

  /// Make cleanup calls.  Retain the stack top list for a repeat call.
  void finalizeAndKeep() {
    assert(!cufs.empty() && "invalid finalize statement context");
    if (cufs.back())
      (*cufs.back())();
  }

  /// Make cleanup calls.  Pop the stack top list.
  void finalizeAndPop() {
    finalizeAndKeep();
    cufs.pop_back();
  }

  /// Make cleanup calls.  Clear the stack top list.
  void finalize() {
    finalizeAndKeep();
    cufs.back().reset();
  }

  bool workListIsEmpty() const {
    return cufs.empty() || llvm::all_of(cufs, [](auto &opt) -> bool {
             return !opt.has_value();
           });
  }

private:
  // A statement context should never be copied or moved.
  StatementContext(const StatementContext &) = delete;
  StatementContext &operator=(const StatementContext &) = delete;
  StatementContext(StatementContext &&) = delete;

  // Stack of cleanup function "lists" (nested cleanup function calls).
  llvm::SmallVector<llvm::Optional<CleanupFunction>> cufs;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_STATEMENTCONTEXT_H
