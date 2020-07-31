//===-- Optimizer/Transforms/Passes.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_TRANSFORMS_PASSES_H
#define OPTIMIZER_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {
class BlockAndValueMapping;
class Operation;
class Pass;
class Region;
} // namespace mlir

namespace fir {

/// Convert fir.select_type to the standard dialect
std::unique_ptr<mlir::Pass> createControlFlowLoweringPass();

/// Effects aware CSE pass
std::unique_ptr<mlir::Pass> createCSEPass();

/// Convert FIR loop constructs to the Affine dialect
std::unique_ptr<mlir::Pass> createPromoteToAffinePass();

/// Convert `fir.do_loop` and `fir.if` to a CFG.  This
/// conversion enables the `createLowerToCFGPass` to transform these to CFG
/// form.
std::unique_ptr<mlir::Pass> createFirToCfgPass();

/// A pass to convert the FIR dialect from "Mem-SSA" form to "Reg-SSA"
/// form. This pass is a port of LLVM's mem2reg pass, but modified for the FIR
/// dialect as well as the restructuring of MLIR's representation to present PHI
/// nodes as block arguments.
std::unique_ptr<mlir::Pass> createMemToRegPass();

/// Support for inlining on FIR.
bool canLegallyInline(mlir::Operation *op, mlir::Region *reg,
                      mlir::BlockAndValueMapping &map);

// declarative passes
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/Transforms/Passes.h.inc"

} // namespace fir

#endif // OPTIMIZER_TRANSFORMS_PASSES_H
