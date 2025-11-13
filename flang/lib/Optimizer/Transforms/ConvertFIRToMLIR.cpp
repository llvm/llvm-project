//===-- ConvertFIRToMLIR.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace fir {
#define GEN_PASS_DEF_CONVERTFIRTOMLIRPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class ConvertFIRToMLIRPass
    : public fir::impl::ConvertFIRToMLIRPassBase<ConvertFIRToMLIRPass> {
public:
  void runOnOperation() override;
};
} // namespace

void ConvertFIRToMLIRPass::runOnOperation() {
  // TODO:
}
