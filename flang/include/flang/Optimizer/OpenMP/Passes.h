//===- Passes.h - OpenMP pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the flang OpenMP passes.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_OPENMP_PASSES_H
#define FORTRAN_OPTIMIZER_OPENMP_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include <memory>

namespace flangomp {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/OpenMP/Passes.h.inc"

} // namespace flangomp

#endif // FORTRAN_OPTIMIZER_OPENMP_PASSES_H
