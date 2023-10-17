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
#include <optional>

namespace clang {
namespace ento {
namespace errno_modeling {

/// Describe how reads and writes of \c errno are handled by the checker.
enum ErrnoCheckState : unsigned {
  /// We do not know anything about 'errno'.
  /// Read and write is always allowed.
  Irrelevant = 0,

  /// Value of 'errno' should be checked to find out if a previous function call
  /// has failed.
  /// When this state is set \c errno must be read by the program before a next
  /// standard function call or other overwrite of \c errno follows, otherwise
  /// a bug report is emitted.
  MustBeChecked = 1,

  /// Value of 'errno' is not allowed to be read, it can contain an unspecified
  /// value.
  /// When this state is set \c errno is not allowed to be read by the program
  /// until it is overwritten or invalidated.
  MustNotBeChecked = 2
};

/// Returns the value of 'errno', if 'errno' was found in the AST.
std::optional<SVal> getErrnoValue(ProgramStateRef State);

/// Returns the errno check state, \c Errno_Irrelevant if 'errno' was not found
/// (this is not the only case for that value).
ErrnoCheckState getErrnoState(ProgramStateRef State);

/// Returns the location that points to the \c MemoryRegion where the 'errno'
/// value is stored. Returns \c std::nullopt if 'errno' was not found. Otherwise
/// it always returns a valid memory region in the system global memory space.
std::optional<Loc> getErrnoLoc(ProgramStateRef State);

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

/// Clear state of errno (make it irrelevant).
ProgramStateRef clearErrnoState(ProgramStateRef State);

/// Determine if a `Decl` node related to 'errno'.
/// This is true if the declaration is the errno variable or a function
/// that returns a pointer to the 'errno' value (usually the 'errno' macro is
/// defined with this function). \p D is not required to be a canonical
/// declaration.
bool isErrno(const Decl *D);

/// Create a NoteTag that displays the message if the 'errno' memory region is
/// marked as interesting, and resets the interestingness.
const NoteTag *getErrnoNoteTag(CheckerContext &C, const std::string &Message);

/// Set errno state for the common case when a standard function is successful.
/// Set \c ErrnoCheckState to \c MustNotBeChecked (the \c errno value is not
/// affected). At the state transition a note tag created by
/// \c getNoteTagForStdSuccess can be used.
ProgramStateRef setErrnoForStdSuccess(ProgramStateRef State, CheckerContext &C);

/// Set errno state for the common case when a standard function fails.
/// Set \c errno value to be not equal to zero and \c ErrnoCheckState to
/// \c Irrelevant . The irrelevant errno state ensures that no related bug
/// report is emitted later and no note tag is needed.
/// \arg \c ErrnoSym Value to be used for \c errno and constrained to be
/// non-zero.
ProgramStateRef setErrnoForStdFailure(ProgramStateRef State, CheckerContext &C,
                                      NonLoc ErrnoSym);

/// Set errno state for the common case when a standard function indicates
/// failure only by \c errno. Sets \c ErrnoCheckState to \c MustBeChecked, and
/// invalidates the errno region (clear of previous value).
/// At the state transition a note tag created by
/// \c getNoteTagForStdMustBeChecked can be used.
/// \arg \c InvalE Expression that causes invalidation of \c errno.
ProgramStateRef setErrnoStdMustBeChecked(ProgramStateRef State,
                                         CheckerContext &C, const Expr *InvalE);

/// Generate the note tag that can be applied at the state generated by
/// \c setErrnoForStdSuccess .
/// \arg \c Fn Name of the (standard) function that is modeled.
const NoteTag *getNoteTagForStdSuccess(CheckerContext &C, llvm::StringRef Fn);

/// Generate the note tag that can be applied at the state generated by
/// \c setErrnoStdMustBeChecked .
/// \arg \c Fn Name of the (standard) function that is modeled.
const NoteTag *getNoteTagForStdMustBeChecked(CheckerContext &C,
                                             llvm::StringRef Fn);

} // namespace errno_modeling
} // namespace ento
} // namespace clang

#endif // LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_ERRNOMODELING_H
