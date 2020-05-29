//===- PassDetail.h - Optimizer Transforms Pass class details ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTMIZER_TRANSFORMS_PASSDETAIL_H_
#define OPTMIZER_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace fir {

#define GEN_PASS_CLASSES
#include "flang/Optimizer/Transforms/Passes.h.inc"

} // end namespace mlir

#endif // OPTMIZER_TRANSFORMS_PASSDETAIL_H_
