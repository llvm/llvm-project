//===- Passes.h - HLFIR pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares HLFIR pass entry points.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_HLFIR_PASSES_H
#define FORTRAN_OPTIMIZER_HLFIR_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace hlfir {
#define GEN_PASS_DECL
#include "flang/Optimizer/HLFIR/Passes.h.inc"

std::unique_ptr<mlir::Pass> createConvertHLFIRtoFIRPass();
std::unique_ptr<mlir::Pass> createBufferizeHLFIRPass();

#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

#endif // FORTRAN_OPTIMIZER_HLFIR_PASSES_H
