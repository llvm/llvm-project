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

#include "SymbolMap.h"
#include "llvm/Support/ErrorHandling.h"
#include <functional>

namespace Fortran::lower {

/// When lowering a statement, large temporaries may be allocated on the heap to
/// buffer intermediate results. These temporaries must be deallocated at the
/// end of the statement. These deallocations are threaded via a
/// StatementContext back to the "end" of the statement.
class StatementContext {
public:
  explicit StatementContext() {
    cleanup = []() {};
  }

  /// Cleanups can be prohibited in some contexts. If prohibited, the compiler
  /// will crash if and where it tries to unexpectedly add a cleanup.
  explicit StatementContext(bool prohibited) : cleanupProhibited{prohibited} {
    cleanup = []() {};
  }

  ~StatementContext() {
    if (!finalized)
      cleanup();
  }

  /// Append the cleanup function `cuf` to the list of cleanups.
  void attachCleanup(std::function<void()> cuf) {
    if (cleanupProhibited)
      llvm::report_fatal_error("expression cleanups disallowed");
    assert(!finalized);
    auto oldCleanup = cleanup;
    cleanup = [=]() {
      cuf();
      oldCleanup();
    };
    cleanupAdded = true;
  }

  /// Force finalization of cleanups. Normally, cleanups are applied by the
  /// destructor, but some statements require the cleanups be added before an Op
  /// that will change the control dependence.
  void finalize() {
    cleanup();
    finalized = true;
    cleanup = []() { llvm::report_fatal_error("already finalized"); };
  }

  bool hasCleanups() const { return cleanupAdded; }

private:
  StatementContext(const StatementContext &) = delete;
  StatementContext &operator=(const StatementContext &) = delete;
  StatementContext(StatementContext &&) = delete;

  std::function<void()> cleanup;
  bool finalized{};
  bool cleanupAdded{};
  bool cleanupProhibited{};
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_STATEMENTCONTEXT_H
