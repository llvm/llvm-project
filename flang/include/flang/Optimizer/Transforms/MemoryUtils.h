//===-- Optimizer/Transforms/MemoryUtils.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
//
// This file defines a utility to replace fir.alloca by dynamic allocation and
// deallocation. The exact kind of dynamic allocation is left to be defined by
// the utility user via callbacks (could be fir.allocmem or custom runtime
// calls).
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_MEMORYUTILS_H
#define FORTRAN_OPTIMIZER_TRANSFORMS_MEMORYUTILS_H

#include "flang/Optimizer/Dialect/FIROps.h"

namespace mlir {
class RewriterBase;
}

namespace fir {

/// Type of callbacks that indicate if a given fir.alloca must be
/// rewritten.
using MustRewriteCallBack = llvm::function_ref<bool(fir::AllocaOp)>;

/// Type of callbacks that produce the replacement for a given fir.alloca.
/// It is provided extra information about the dominance of the deallocation
/// points that have been identified, and may refuse to replace the alloca,
/// even if the MustRewriteCallBack previously returned true, in which case
/// it should return a null value.
/// The callback should not delete the alloca, the utility will do it.
using AllocaRewriterCallBack = llvm::function_ref<mlir::Value(
    mlir::OpBuilder &, fir::AllocaOp, bool allocaDominatesDeallocLocations)>;
/// Type of callbacks that must generate deallocation of storage obtained via
/// AllocaRewriterCallBack calls.
using DeallocCallBack =
    llvm::function_ref<void(mlir::Location, mlir::OpBuilder &, mlir::Value)>;

/// Utility to replace fir.alloca by dynamic allocations inside \p parentOp.
/// \p MustRewriteCallBack lets the user control which fir.alloca should be
/// replaced. \p AllocaRewriterCallBack lets the user define how the new memory
/// should be allocated. \p DeallocCallBack lets the user decide how the memory
/// should be deallocated. The boolean result indicates if the utility succeeded
/// to replace all fir.alloca as requested by the user. Causes of failures are
/// the presence of unregistered operations, or OpenMP/ACC recipe operations
/// that return memory allocated inside their region.
bool replaceAllocas(mlir::RewriterBase &rewriter, mlir::Operation *parentOp,
                    MustRewriteCallBack, AllocaRewriterCallBack,
                    DeallocCallBack);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_MEMORYUTILS_H
