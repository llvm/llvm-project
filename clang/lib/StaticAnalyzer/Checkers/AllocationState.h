//===--- AllocationState.h ------------------------------------- *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_ALLOCATIONSTATE_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_ALLOCATIONSTATE_H

#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"

namespace clang {
namespace ento {

namespace allocation_state {

ProgramStateRef markReleased(ProgramStateRef State, SymbolRef Sym,
                             const Expr *Origin);

/// Returns true if \p Sym is currently tracked by MallocChecker as a
/// \c new-allocated pointer that has escaped into an opaque owner (e.g., was
/// stored inside a \c std::unique_ptr). This is the precise precondition for
/// calling \c transferToCallerNew: we only reclaim ownership when we have
/// definitive evidence that MallocChecker originally allocated the symbol.
bool isReleasedByNew(ProgramStateRef State, SymbolRef Sym);

/// Transitions \p Sym from the \c Escaped state back to the \c Allocated
/// state with family \c AF_CXXNew. Use this when \c unique_ptr::release()
/// transfers ownership of a \c new-allocated object back to the caller. After
/// this call, MallocChecker will report a memory leak if \p Sym is not freed.
///
/// Precondition: \c isReleasedByNew(State, Sym) must return \c true.
ProgramStateRef transferToCallerNew(ProgramStateRef State, SymbolRef Sym,
                                    const Expr *Origin);

/// This function provides an additional visitor that augments the bug report
/// with information relevant to memory errors caused by the misuse of
/// AF_InnerBuffer symbols.
std::unique_ptr<BugReporterVisitor> getInnerPointerBRVisitor(SymbolRef Sym);

/// 'Sym' represents a pointer to the inner buffer of a container object.
/// This function looks up the memory region of that object in
/// DanglingInternalBufferChecker's program state map.
const MemRegion *getContainerObjRegion(ProgramStateRef State, SymbolRef Sym);

} // end namespace allocation_state

} // end namespace ento
} // end namespace clang

#endif
