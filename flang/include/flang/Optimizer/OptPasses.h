//===-- Optimizer/Transforms/Passes.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_OPTPASSES_H
#define OPTIMIZER_OPTPASSES_H

#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Transforms/Passes.h"

namespace fir {
inline void registerOptPasses() {
  registerOptCodeGenPasses();
  registerOptTransformPasses();
}
} // namespace fir

#endif // OPTIMIZER_OPTPASSES_H
