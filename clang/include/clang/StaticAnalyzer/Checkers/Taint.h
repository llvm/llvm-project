//=== Taint.h - Taint tracking and basic propagation rules. --------*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines basic, non-domain-specific mechanisms for tracking tainted values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_TAINT_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_TAINT_H

#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/Support/Compiler.h"

namespace clang {
namespace ento {
namespace taint {

/// The type of taint, which helps to differentiate between different types of
/// taint.
using TaintTagType = unsigned;

static constexpr TaintTagType TaintTagGeneric = 0;

/// Create a new state in which the value of the statement is marked as tainted.
[[nodiscard]] ProgramStateRef CLANG_ABI addTaint(ProgramStateRef State, const Stmt *S,
                                       const LocationContext *LCtx,
                                       TaintTagType Kind = TaintTagGeneric);

/// Create a new state in which the value is marked as tainted.
[[nodiscard]] ProgramStateRef CLANG_ABI addTaint(ProgramStateRef State, SVal V,
                                       TaintTagType Kind = TaintTagGeneric);

/// Create a new state in which the symbol is marked as tainted.
[[nodiscard]] ProgramStateRef CLANG_ABI addTaint(ProgramStateRef State, SymbolRef Sym,
                                       TaintTagType Kind = TaintTagGeneric);

/// Create a new state in which the pointer represented by the region
/// is marked as tainted.
[[nodiscard]] ProgramStateRef CLANG_ABI addTaint(ProgramStateRef State,
                                       const MemRegion *R,
                                       TaintTagType Kind = TaintTagGeneric);

[[nodiscard]] ProgramStateRef CLANG_ABI removeTaint(ProgramStateRef State, SVal V);

[[nodiscard]] ProgramStateRef CLANG_ABI removeTaint(ProgramStateRef State,
                                          const MemRegion *R);

[[nodiscard]] ProgramStateRef CLANG_ABI removeTaint(ProgramStateRef State, SymbolRef Sym);

/// Create a new state in a which a sub-region of a given symbol is tainted.
/// This might be necessary when referring to regions that can not have an
/// individual symbol, e.g. if they are represented by the default binding of
/// a LazyCompoundVal.
[[nodiscard]] ProgramStateRef
CLANG_ABI addPartialTaint(ProgramStateRef State, SymbolRef ParentSym,
                const SubRegion *SubRegion,
                TaintTagType Kind = TaintTagGeneric);

/// Check if the statement has a tainted value in the given state.
CLANG_ABI bool isTainted(ProgramStateRef State, const Stmt *S,
               const LocationContext *LCtx,
               TaintTagType Kind = TaintTagGeneric);

/// Check if the value is tainted in the given state.
CLANG_ABI bool isTainted(ProgramStateRef State, SVal V,
               TaintTagType Kind = TaintTagGeneric);

/// Check if the symbol is tainted in the given state.
CLANG_ABI bool isTainted(ProgramStateRef State, SymbolRef Sym,
               TaintTagType Kind = TaintTagGeneric);

/// Check if the pointer represented by the region is tainted in the given
/// state.
CLANG_ABI bool isTainted(ProgramStateRef State, const MemRegion *Reg,
               TaintTagType Kind = TaintTagGeneric);

/// Returns the tainted Symbols for a given Statement and state.
CLANG_ABI std::vector<SymbolRef> getTaintedSymbols(ProgramStateRef State, const Stmt *S,
                                         const LocationContext *LCtx,
                                         TaintTagType Kind = TaintTagGeneric);

/// Returns the tainted Symbols for a given SVal and state.
CLANG_ABI std::vector<SymbolRef> getTaintedSymbols(ProgramStateRef State, SVal V,
                                         TaintTagType Kind = TaintTagGeneric);

/// Returns the tainted Symbols for a SymbolRef and state.
CLANG_ABI std::vector<SymbolRef> getTaintedSymbols(ProgramStateRef State, SymbolRef Sym,
                                         TaintTagType Kind = TaintTagGeneric);

/// Returns the tainted (index, super/sub region, symbolic region) symbols
/// for a given memory region.
CLANG_ABI std::vector<SymbolRef> getTaintedSymbols(ProgramStateRef State,
                                         const MemRegion *Reg,
                                         TaintTagType Kind = TaintTagGeneric);

CLANG_ABI std::vector<SymbolRef> getTaintedSymbolsImpl(ProgramStateRef State,
                                             const Stmt *S,
                                             const LocationContext *LCtx,
                                             TaintTagType Kind,
                                             bool returnFirstOnly);

CLANG_ABI std::vector<SymbolRef> getTaintedSymbolsImpl(ProgramStateRef State, SVal V,
                                             TaintTagType Kind,
                                             bool returnFirstOnly);

CLANG_ABI std::vector<SymbolRef> getTaintedSymbolsImpl(ProgramStateRef State,
                                             SymbolRef Sym, TaintTagType Kind,
                                             bool returnFirstOnly);

CLANG_ABI std::vector<SymbolRef> getTaintedSymbolsImpl(ProgramStateRef State,
                                             const MemRegion *Reg,
                                             TaintTagType Kind,
                                             bool returnFirstOnly);

CLANG_ABI void printTaint(ProgramStateRef State, raw_ostream &Out, const char *nl = "\n",
                const char *sep = "");

LLVM_DUMP_METHOD void CLANG_ABI dumpTaint(ProgramStateRef State);
} // namespace taint
} // namespace ento
} // namespace clang

#endif
