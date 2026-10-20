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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <optional>

namespace mlir {
class Location;
class Region;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace Fortran::lower {

/// When lowering a statement, temporaries for intermediate results may be
/// allocated on the heap. A StatementContext enables their deallocation
/// with one of several explicit finalize calls, or with an implicit
/// call to finalizeAndPop() at the end of the context. A context may prohibit
/// temporary allocation. Otherwise, an initial "outer" context scope may have
/// nested context scopes, which must make explicit subscope finalize calls.
///
/// In addition to being useful for individual action statement contexts, a
/// StatementContext is also useful for construct blocks delimited by a pair
/// of statements such as (block-stmt, end-block-stmt), or a program unit
/// delimited by a pair of statements such as (subroutine-stmt, end-subroutine-
/// stmt). Attached cleanup code for these contexts may include stack
/// management code, deallocation code, and finalization of derived type
/// entities in the context.
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

  /// Make cleanup calls. Retain the stack top list for a repeat call.
  void finalizeAndKeep() {
    assert(!cufs.empty() && "invalid finalize statement context");
    if (cufs.back())
      (*cufs.back())();
  }

  /// Make cleanup calls. Clear the stack top list.
  void finalizeAndReset() {
    finalizeAndKeep();
    cufs.back().reset();
  }

  /// Pop the stack top list.
  void pop() { cufs.pop_back(); }

  /// Make cleanup calls. Pop the stack top list.
  void finalizeAndPop() {
    finalizeAndKeep();
    pop();
  }

  bool hasCode() const {
    return !cufs.empty() && llvm::any_of(cufs, [](auto &opt) -> bool {
      return opt.has_value();
    });
  }

private:
  // A statement context should never be copied or moved.
  StatementContext(const StatementContext &) = delete;
  StatementContext &operator=(const StatementContext &) = delete;
  StatementContext(StatementContext &&) = delete;

  // Stack of cleanup function "lists" (nested cleanup function calls).
  llvm::SmallVector<std::optional<CleanupFunction>> cufs;
};

/// If \p context contains any cleanups, ensure \p region has a block, and
/// generate the cleanup inside that block.
void genCleanUpInRegionIfAny(mlir::Location loc, fir::FirOpBuilder &builder,
                             mlir::Region &region, StatementContext &context);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_STATEMENTCONTEXT_H
