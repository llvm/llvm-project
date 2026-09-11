//===-- Assign.h - generate assignment runtime API calls --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ASSIGN_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ASSIGN_H

namespace mlir {
class Value;
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

/// Generate runtime call to assign \p sourceBox to \p destBox.
/// \p destBox must be a fir.ref<fir.box<T>> and \p sourceBox a fir.box<T>.
/// \p destBox Fortran descriptor may be modified if destBox is an allocatable
/// according to Fortran allocatable assignment rules, otherwise it is not
/// modified.
void genAssign(fir::FirOpBuilder &builder, mlir::Location loc,
               mlir::Value destBox, mlir::Value sourceBox);

/// Generate runtime call to AssignPolymorphic \p sourceBox to \p destBox.
/// \p destBox must be a fir.ref<fir.box<T>> and \p sourceBox a fir.box<T>.
/// \p destBox Fortran descriptor may be modified if destBox is an allocatable
/// according to Fortran allocatable assignment rules.
void genAssignPolymorphic(fir::FirOpBuilder &builder, mlir::Location loc,
                          mlir::Value destBox, mlir::Value sourceBox);

/// Generate runtime call to AssignExplicitLengthCharacter to assign
/// \p sourceBox to \p destBox where \p destBox is a whole allocatable character
/// with explicit or assumed length. After the assignment, the length of
/// \p destBox will remain what it was, even if allocation or reallocation
/// occurred. For assignments to a whole allocatable with deferred length,
/// genAssign should be used.
/// \p destBox must be a fir.ref<fir.box<T>> and \p sourceBox a fir.box<T>.
/// \p destBox Fortran descriptor may be modified if destBox is an allocatable
/// according to Fortran allocatable assignment rules.
void genAssignExplicitLengthCharacter(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value destBox,
                                      mlir::Value sourceBox);

/// Generate runtime call to assign \p sourceBox to \p destBox.
/// \p destBox must be a fir.ref<fir.box<T>> and \p sourceBox a fir.box<T>.
/// \p destBox Fortran descriptor may be modified if destBox is an allocatable
/// according to Fortran allocatable assignment rules, otherwise it is not
/// modified.
void genAssignTemporary(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::Value destBox, mlir::Value sourceBox);

/// Generate runtime call to "CopyInAssign" runtime API.
void genCopyInAssign(fir::FirOpBuilder &builder, mlir::Location loc,
                     mlir::Value tempBoxAddr, mlir::Value varBoxAddr);
/// Generate runtime call to "CopyOutAssign" runtime API.
void genCopyOutAssign(fir::FirOpBuilder &builder, mlir::Location loc,
                      mlir::Value varBoxAddr, mlir::Value tempBoxAddr);

/// Generate runtime call to "CopyOutReadOnlyMode": returns an i32 mode value
/// (0 = off, 1 = trust the memory-map snapshot, 2 = confirm each hit) for the
/// read-only copy-out skip policy, so inlined copy-out code can apply the
/// same policy as the CopyOutAssign runtime.
mlir::Value genCopyOutReadOnlyMode(fir::FirOpBuilder &builder,
                                   mlir::Location loc);
/// Generate runtime call to "CopyOutReadOnlyCandidate": i1, true iff \p box's
/// data span lies within the snapshot's read-only regions (no system calls).
mlir::Value genCopyOutReadOnlyCandidate(fir::FirOpBuilder &builder,
                                        mlir::Location loc, mlir::Value box);
/// Generate runtime call to "CopyOutReadOnlyConfirm": i1, true iff \p box's
/// data span is read-only in the current memory map (mode-2 re-confirmation).
mlir::Value genCopyOutReadOnlyConfirm(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value box);
/// Generate runtime call to "NoteSkippedCopyOut" (diagnostics for a skipped
/// copy-out).
void genNoteSkippedCopyOut(fir::FirOpBuilder &builder, mlir::Location loc);

/// Generate runtime call to AssignSimple (fast path for intrinsic types).
/// \p destBox must be a fir.ref<fir.box<T>> and \p sourceBox a fir.box<T>.
/// Preconditions enforced at call site:
///   - Intrinsic element type (integer, real, complex, logical)
///   - Matching ranks (no scalar-to-array broadcasting)
///   - Same element byte size
///   - Non-volatile
/// Runtime handles: contiguous and non-contiguous layouts, aliasing detection,
/// allocatable reallocation.
void genAssignSimple(fir::FirOpBuilder &builder, mlir::Location loc,
                     mlir::Value destBox, mlir::Value sourceBox);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ASSIGN_H
