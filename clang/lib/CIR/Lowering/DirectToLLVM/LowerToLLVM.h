//====- LowerToLLVM.h- Lowering from CIR to LLVM --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an interface for converting CIR modules to LLVM IR.
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_CIR_LOWERTOLLVM_H
#define CLANG_CIR_LOWERTOLLVM_H

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace cir {

namespace direct {

/// Convert a CIR attribute to an LLVM attribute. May use the datalayout for
/// lowering attributes to-be-stored in memory.
mlir::Value lowerCirAttrAsValue(mlir::Operation *parentOp, mlir::Attribute attr,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *converter);

mlir::LLVM::Linkage convertLinkage(cir::GlobalLinkageKind linkage);

void convertSideEffectForCall(mlir::Operation *callOp, bool isNothrow,
                              cir::SideEffect sideEffect,
                              mlir::LLVM::MemoryEffectsAttr &memoryEffect,
                              bool &noUnwind, bool &willReturn);

#define GET_LLVM_LOWERING_PATTERNS
#include "clang/CIR/Dialect/IR/CIRLowering.inc"
#undef GET_LLVM_LOWERING_PATTERNS

} // namespace direct
} // namespace cir

#endif // CLANG_CIR_LOWERTOLLVM_H
