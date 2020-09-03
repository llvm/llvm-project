//===-- Inliner.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<bool>
    aggressivelyInline("inline-all",
                       llvm::cl::desc("aggressively inline everything"),
                       llvm::cl::init(false));

/// Should we inline the callable `op` into region `reg`?
bool fir::canLegallyInline(mlir::Operation *op, mlir::Region *reg,
                           mlir::BlockAndValueMapping &map) {
  return aggressivelyInline;
}
