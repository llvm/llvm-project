//===-- Optimizer/Transforms/Passes.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H
#define FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H

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

//===----------------------------------------------------------------------===//
// Passes defined in Passes.td
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createAbstractResultOptPass();
std::unique_ptr<mlir::Pass> createAffineDemotionPass();
std::unique_ptr<mlir::Pass> createArrayValueCopyPass();
std::unique_ptr<mlir::Pass> createFirToCfgPass();
std::unique_ptr<mlir::Pass> createCharacterConversionPass();
std::unique_ptr<mlir::Pass> createExternalNameConversionPass();
std::unique_ptr<mlir::Pass> createMemDataFlowOptPass();
std::unique_ptr<mlir::Pass> createPromoteToAffinePass();
std::unique_ptr<mlir::Pass> createMemoryAllocationPass();
std::unique_ptr<mlir::Pass>
createMemoryAllocationPass(bool dynOnHeap, std::size_t maxStackSize);

// declarative passes
#define GEN_PASS_REGISTRATION
#include "flang/Optimizer/Transforms/Passes.h.inc"

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_PASSES_H
