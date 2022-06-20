//=== ErrnoModeling.h - Tracking value of 'errno'. -----------------*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines inter-checker API for using the system value 'errno'.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_ERRNOMODELING_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_ERRNOMODELING_H

#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"

namespace clang {
namespace ento {
namespace errno_modeling {

enum ErrnoCheckState : unsigned {
  /// We do not know anything about 'errno'.
  Irrelevant = 0,

  /// Value of 'errno' should be checked to find out if a previous function call
  /// has failed.
  MustBeChecked = 1,

  /// Value of 'errno' is not allowed to be read, it can contain an unspecified
  /// value.
  MustNotBeChecked = 2
};

/// Returns the value of 'errno', if 'errno' was found in the AST.
llvm::Optional<SVal> getErrnoValue(ProgramStateRef State);

/// Returns the errno check state, \c Errno_Irrelevant if 'errno' was not found
/// (this is not the only case for that value).
ErrnoCheckState getErrnoState(ProgramStateRef State);

/// Returns the location that points to the \c MemoryRegion where the 'errno'
/// value is stored. Returns \c None if 'errno' was not found. Otherwise it
/// always returns a valid memory region in the system global memory space.
llvm::Optional<Loc> getErrnoLoc(ProgramStateRef State);

/// Set value of 'errno' to any SVal, if possible.
/// The errno check state is set always when the 'errno' value is set.
ProgramStateRef setErrnoValue(ProgramStateRef State,
                              const LocationContext *LCtx, SVal Value,
                              ErrnoCheckState EState);

/// Set value of 'errno' to a concrete (signed) integer, if possible.
/// The errno check state is set always when the 'errno' value is set.
ProgramStateRef setErrnoValue(ProgramStateRef State, CheckerContext &C,
                              uint64_t Value, ErrnoCheckState EState);

/// Set the errno check state, do not modify the errno value.
ProgramStateRef setErrnoState(ProgramStateRef State, ErrnoCheckState EState);

/// Determine if a `Decl` node related to 'errno'.
/// This is true if the declaration is the errno variable or a function
/// that returns a pointer to the 'errno' value (usually the 'errno' macro is
/// defined with this function). \p D is not required to be a canonical
/// declaration.
bool isErrno(const Decl *D);

/// Create a NoteTag that displays the message if the 'errno' memory region is
/// marked as interesting, and resets the interestingness.
const NoteTag *getErrnoNoteTag(CheckerContext &C, const std::string &Message);

} // namespace errno_modeling
} // namespace ento
} // namespace clang

#endif // LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_ERRNOMODELING_H
