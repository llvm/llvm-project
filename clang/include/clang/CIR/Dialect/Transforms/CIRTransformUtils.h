//===- CIRTransformUtils.h - Shared helpers for CIR transforms -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRTRANSFORMUTILS_H
#define LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRTRANSFORMUTILS_H

#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "llvm/ADT/SmallVector.h"

namespace cir {

/// Replace a `cir::CallOp` with a `cir::TryCallOp` whose unwind destination
/// is \p unwindDest. The call's parent block is split immediately after the
/// call; the resulting suffix block becomes the try_call's normal
/// destination and is returned to the caller.
///
/// All attributes of the original call other than the callee and operand
/// segment sizes (which `TryCallOp::create` sets itself) are copied onto
/// the new try_call. Uses of the original call's result, if any, are
/// redirected to the try_call's result, and the original call is erased.
///
/// The call must not already be marked nothrow.
mlir::Block *replaceCallWithTryCall(cir::CallOp callOp, mlir::Block *unwindDest,
                                    mlir::Location loc,
                                    mlir::RewriterBase &rewriter);

/// Collect ops in blocks that are unreachable from their region's entry,
/// appending them to \p ops. Used by CIR passes that drive
/// `applyPartialConversion` and need to feed it operations the conversion
/// driver's dominance-order traversal would otherwise skip.
void collectUnreachable(mlir::Operation *parent,
                        llvm::SmallVectorImpl<mlir::Operation *> &ops);

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_TRANSFORMS_CIRTRANSFORMUTILS_H
