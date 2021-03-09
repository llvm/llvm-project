//===-- Optimizer/OptPasses.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_OPTPASSES_H
#define FORTRAN_OPTIMIZER_OPTPASSES_H

#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Transforms/Passes.h"

namespace fir {
/// Register the passes in the flang/Optimizer directory.
/// TODO: Consider merging the registration of all passes in 1 function.
inline void registerOptimizerPasses() {
  registerOptCodeGenPasses();
  registerOptTransformPasses();
}
} // namespace fir

#endif // FORTRAN_OPTIMIZER_OPTPASSES_H
