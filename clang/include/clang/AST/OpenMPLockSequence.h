//===- OpenMPLockSequence.h - OpenMP lock sequences in a block --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Where the threads of a wavefront run in lockstep, a thread that takes an
// OpenMP lock cannot release it while another thread of its wavefront spins for
// the same lock. A compound statement containing
//
//   omp_set_lock(L); ... omp_unset_lock(L);
//
// can instead be run by the threads of a wavefront one at a time, provided that
// each thread releases every lock it takes before the next thread's turn. The
// analysis in this file is used both by code generation and semantic analysis;
// the latter warns about omp_set_lock cases that cannot be handled like this.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_OPENMPLOCKSEQUENCE_H
#define LLVM_CLANG_AST_OPENMPLOCKSEQUENCE_H

#include "clang/Basic/LLVM.h"

namespace clang {

class ASTContext;
class CallExpr;
class Decl;
class Stmt;

/// Reasons why an omp_set_lock ... omp_unset_lock sequence might be rejected.
/// Keep this list aligned with the %select in
/// warn_omp_lock_not_lane_serialized.
enum class OMPLockSequenceRejection : unsigned {
  /// No matching omp_unset_lock follows it in the same compound statement.
  NoUnset,
  /// Control can enter or leave the code before the matching omp_unset_lock
  /// other than at its ends.
  Escape,
  /// A lock routine is used before the matching omp_unset_lock other than in
  /// properly nested pairs at the top level of the compound statement.
  OtherLockUse,
  /// An OpenMP directive comes before the matching omp_unset_lock.
  Directive,
  /// A variable is declared before the matching omp_unset_lock.
  Declaration,
  /// The lock expression has side effects.
  SideEffects,
  /// The omp_set_lock statement is labelled.
  Labelled,
  /// The omp_set_lock call is not a statement of a compound statement.
  NotInBlock,
};

/// The result of looking for a lock sequence at the start of some statements.
struct OMPLockSequence {
  /// The omp_set_lock call made by the first statement, if it makes one.
  const CallExpr *SetCall = nullptr;
  /// The number of statements in the sequence, or 0 if there is none.
  unsigned Length = 0;
  /// Why there is no sequence, if SetCall is set and Length is 0.
  OMPLockSequenceRejection Rejection = OMPLockSequenceRejection::NoUnset;
};

/// Returns true if \p D is one of the OpenMP lock routines that set, unset or
/// test a lock.
bool isOMPLockRoutine(const Decl *D);

/// Returns true if \p CE calls omp_set_lock.
bool isOMPSetLockCall(const CallExpr *CE);

/// Looks for a sequence `omp_set_lock(L); ... omp_unset_lock(L);` that the
/// threads of a warp can run one at a time at the start of \p Stmts, the
/// statements of a compound statement not yet considered. The sequence must use
/// the same side-effect-free lock expression in both calls; the statements in
/// between must be enterable only at the start and leavable only at the end,
/// declare nothing, contain no OpenMP directive, and use the lock routines only
/// in further such pairs, properly nested, among themselves.
OMPLockSequence matchOMPLockSequence(ArrayRef<Stmt *> Stmts,
                                     const ASTContext &Ctx);

} // namespace clang

#endif // LLVM_CLANG_AST_OPENMPLOCKSEQUENCE_H
