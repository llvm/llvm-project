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
#include "clang/CIR/Interfaces/CIROpInterfaces.h"

namespace cir {

namespace direct {

struct LLVMBlockAddressInfo;

/// Convert a CIR attribute to an LLVM attribute. May use the datalayout for
/// lowering attributes to-be-stored in memory. When the attribute may contain
/// block address attributes, `blockInfoAddr` is used to resolve them.
mlir::Value lowerCirAttrAsValue(mlir::Operation *parentOp, mlir::Attribute attr,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *converter,
                                LLVMBlockAddressInfo *blockInfoAddr = nullptr);

mlir::LLVM::Linkage convertLinkage(cir::GlobalLinkageKind linkage);

void convertSideEffectForCall(mlir::Operation *callOp, bool isNothrow,
                              cir::SideEffect sideEffect,
                              mlir::LLVM::MemoryEffectsAttr &memoryEffect,
                              bool &noUnwind, bool &willReturn, bool &noReturn);

struct LLVMBlockAddressInfo {
  // Get the next tag index
  uint32_t getTagIndex() { return blockTagOpIndex++; }

  void mapBlockTag(cir::BlockAddrInfoAttr info, mlir::LLVM::BlockTagOp tagOp) {
    [[maybe_unused]] auto result = blockInfoToTagOp.try_emplace(info, tagOp);
    assert(result.second &&
           "attempting to map a BlockTag operation that is already mapped");
  }

  // Lookup a BlockTagOp, may return nullptr if not yet registered.
  mlir::LLVM::BlockTagOp lookupBlockTag(cir::BlockAddrInfoAttr info) const {
    return blockInfoToTagOp.lookup(info);
  }

  // Record an unresolved BlockAddressOp that needs patching later.
  void addUnresolvedBlockAddress(mlir::LLVM::BlockAddressOp op,
                                 cir::BlockAddrInfoAttr info) {
    unresolvedBlockAddressOp.try_emplace(op, info);
  }

  void clearUnresolvedMap() { unresolvedBlockAddressOp.clear(); }

  llvm::DenseMap<mlir::LLVM::BlockAddressOp, cir::BlockAddrInfoAttr> &
  getUnresolvedBlockAddress() {
    return unresolvedBlockAddressOp;
  }

private:
  // Maps a (function name, label name) pair to the corresponding BlockTagOp.
  // Used to resolve CIR LabelOps into their LLVM BlockTagOp.
  llvm::DenseMap<cir::BlockAddrInfoAttr, mlir::LLVM::BlockTagOp>
      blockInfoToTagOp;
  // Tracks BlockAddressOps that could not yet be fully resolved because
  // their BlockTagOp was not available at the time of lowering. The map
  // stores the unresolved BlockAddressOp along with its (function name, label
  // name) pair so it can be patched later.
  llvm::DenseMap<mlir::LLVM::BlockAddressOp, cir::BlockAddrInfoAttr>
      unresolvedBlockAddressOp;
  int32_t blockTagOpIndex;
};

// Lower a floating-point operation with an fenv attribute to a call to the
// matching experimental constrained floating-point intrinsic. The value
// operands are followed by metadata operands for the rounding mode (only when
// `hasRoundingMode` is set) and the exception behavior. Used by the hand-written
// lowering patterns whose no-fenv path needs custom handling (e.g. fmaxnum).
mlir::LogicalResult lowerToConstrainedFPIntrinsic(
    mlir::Operation *op, mlir::ValueRange operands, cir::FenvAttr fenv,
    mlir::Type llvmResTy, mlir::ConversionPatternRewriter &rewriter,
    llvm::StringRef constrainedMnemonic, bool hasRoundingMode);

// Shared lowering for floating-point operations that carry an optional `fenv`
// attribute. Without the attribute, the operation is lowered to the plain LLVM
// operation `LLVMOp`. With the attribute, it is lowered to a call to the
// matching experimental constrained floating-point intrinsic (identified by
// `constrainedMnemonic`) using the generic `llvm.call_intrinsic` form. It
// handles both unary and binary operations and is invoked by the table-generated
// LLVM lowering patterns below.
template <typename LLVMOp>
mlir::LogicalResult
lowerConstrainableFPOp(mlir::Operation *op, mlir::ValueRange operands,
                       cir::FenvAttr fenv,
                       const mlir::TypeConverter &typeConverter,
                       mlir::ConversionPatternRewriter &rewriter,
                       llvm::StringRef constrainedMnemonic,
                       bool hasRoundingMode);

#define GET_LLVM_LOWERING_PATTERNS
#include "clang/CIR/Dialect/IR/CIRLowering.inc"
#undef GET_LLVM_LOWERING_PATTERNS

} // namespace direct
} // namespace cir

#endif // CLANG_CIR_LOWERTOLLVM_H
