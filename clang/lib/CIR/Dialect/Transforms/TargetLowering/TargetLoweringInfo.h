//===---- TargetLoweringInfo.h - Encapsulate target details -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics the TargetCodeGenInfo class from the file
// clang/lib/CodeGen/TargetInfo.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_TARGETLOWERINGINFO_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_TARGETLOWERINGINFO_H

#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include <memory>
#include <string>

namespace cir {

class VAArgOp;

class TargetLoweringInfo {
public:
  virtual ~TargetLoweringInfo();

  virtual cir::SyncScopeKind
  convertSyncScope(cir::SyncScopeKind syncScope) const;

  virtual unsigned
  getTargetAddrSpaceFromCIRAddrSpace(cir::LangAddressSpace addrSpace) const {
    return 0;
  };

  /// Expand an aggregate `cir.va_arg` into the target's variadic
  /// register-save-area sequence and return the produced value.  `valist` is
  /// the (ABI-converted) `va_list` pointer operand.  Returns a null value on
  /// failure, after emitting a not-yet-implemented diagnostic on `op`.  Scalar
  /// `cir.va_arg` is lowered generically and never reaches this hook.  The
  /// base implementation reports not-yet-implemented for every target; only
  /// targets with a specialized subclass expand the aggregate.
  virtual mlir::Value
  lowerAggregateVAArg(CIRBaseBuilderTy &builder, cir::VAArgOp op,
                      mlir::Value valist,
                      const cir::CIRDataLayout &dataLayout) const;
};

// Target-specific factory functions.
std::unique_ptr<TargetLoweringInfo> createAMDGPUTargetLoweringInfo();

std::unique_ptr<TargetLoweringInfo> createNVPTXTargetLoweringInfo();

std::unique_ptr<TargetLoweringInfo> createSPIRVTargetLoweringInfo();

std::unique_ptr<TargetLoweringInfo> createX86_64TargetLoweringInfo();

} // namespace cir

#endif
