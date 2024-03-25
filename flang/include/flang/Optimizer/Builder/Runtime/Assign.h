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

/// Generate runtime call to CopyOutAssign to assign \p sourceBox to
/// \p destBox. This call implements the copy-out of a temporary
/// (\p sourceBox) to the actual argument (\p destBox) passed to a procedure,
/// after the procedure returns to the caller.
/// If \p skipToInit is false, then \p destBox will be initialized before
/// the assignment, otherwise, it is assumed to be already initialized.
/// The runtime makes sure that there is no reallocation of the top-level
/// entity represented by \p destBox. If reallocation is required
/// for the components of \p destBox, then it is done without finalization.
void genCopyOutAssign(fir::FirOpBuilder &builder, mlir::Location loc,
                      mlir::Value destBox, mlir::Value sourceBox,
                      bool skipToInit);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ASSIGN_H
