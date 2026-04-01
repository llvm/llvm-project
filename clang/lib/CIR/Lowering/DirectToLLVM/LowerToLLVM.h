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

#include "aiir/Dialect/LLVMIR/LLVMAttrs.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"

namespace cir {

namespace direct {

/// Convert a CIR attribute to an LLVM attribute. May use the datalayout for
/// lowering attributes to-be-stored in memory.
aiir::Value lowerCirAttrAsValue(aiir::Operation *parentOp, aiir::Attribute attr,
                                aiir::ConversionPatternRewriter &rewriter,
                                const aiir::TypeConverter *converter);

aiir::LLVM::Linkage convertLinkage(cir::GlobalLinkageKind linkage);

void convertSideEffectForCall(aiir::Operation *callOp, bool isNothrow,
                              cir::SideEffect sideEffect,
                              aiir::LLVM::MemoryEffectsAttr &memoryEffect,
                              bool &noUnwind, bool &willReturn, bool &noReturn);

struct LLVMBlockAddressInfo {
  // Get the next tag index
  uint32_t getTagIndex() { return blockTagOpIndex++; }

  void mapBlockTag(cir::BlockAddrInfoAttr info, aiir::LLVM::BlockTagOp tagOp) {
    auto result = blockInfoToTagOp.try_emplace(info, tagOp);
    assert(result.second &&
           "attempting to map a BlockTag operation that is already mapped");
  }

  // Lookup a BlockTagOp, may return nullptr if not yet registered.
  aiir::LLVM::BlockTagOp lookupBlockTag(cir::BlockAddrInfoAttr info) const {
    return blockInfoToTagOp.lookup(info);
  }

  // Record an unresolved BlockAddressOp that needs patching later.
  void addUnresolvedBlockAddress(aiir::LLVM::BlockAddressOp op,
                                 cir::BlockAddrInfoAttr info) {
    unresolvedBlockAddressOp.try_emplace(op, info);
  }

  void clearUnresolvedMap() { unresolvedBlockAddressOp.clear(); }

  llvm::DenseMap<aiir::LLVM::BlockAddressOp, cir::BlockAddrInfoAttr> &
  getUnresolvedBlockAddress() {
    return unresolvedBlockAddressOp;
  }

private:
  // Maps a (function name, label name) pair to the corresponding BlockTagOp.
  // Used to resolve CIR LabelOps into their LLVM BlockTagOp.
  llvm::DenseMap<cir::BlockAddrInfoAttr, aiir::LLVM::BlockTagOp>
      blockInfoToTagOp;
  // Tracks BlockAddressOps that could not yet be fully resolved because
  // their BlockTagOp was not available at the time of lowering. The map
  // stores the unresolved BlockAddressOp along with its (function name, label
  // name) pair so it can be patched later.
  llvm::DenseMap<aiir::LLVM::BlockAddressOp, cir::BlockAddrInfoAttr>
      unresolvedBlockAddressOp;
  int32_t blockTagOpIndex;
};

#define GET_LLVM_LOWERING_PATTERNS
#include "clang/CIR/Dialect/IR/CIRLowering.inc"
#undef GET_LLVM_LOWERING_PATTERNS

} // namespace direct
} // namespace cir

#endif // CLANG_CIR_LOWERTOLLVM_H
