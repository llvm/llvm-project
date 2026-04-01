//===-- Trampoline.h - Runtime trampoline pool builder ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder routines for generating calls to the Fortran runtime trampoline
// pool APIs (_FortranATrampolineInit, _FortranATrampolineAdjust,
// _FortranATrampolineFree).
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_TRAMPOLINE_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_TRAMPOLINE_H

namespace aiir {
class Value;
class Location;
} // namespace aiir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

/// Generate a call to _FortranATrampolineInit.
/// Returns an opaque handle (void*) for the trampoline.
aiir::Value genTrampolineInit(fir::FirOpBuilder &builder, aiir::Location loc,
                              aiir::Value scratch, aiir::Value calleeAddress,
                              aiir::Value staticChainAddress);

/// Generate a call to _FortranATrampolineAdjust.
/// Returns the callable function pointer for the trampoline.
aiir::Value genTrampolineAdjust(fir::FirOpBuilder &builder, aiir::Location loc,
                                aiir::Value handle);

/// Generate a call to _FortranATrampolineFree.
/// Frees the trampoline slot.
void genTrampolineFree(fir::FirOpBuilder &builder, aiir::Location loc,
                       aiir::Value handle);

} // namespace fir::runtime

#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_TRAMPOLINE_H
