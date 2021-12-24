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

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ASSIGN_H
