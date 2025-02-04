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

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ASSIGN_H
