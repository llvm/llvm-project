//===---- TargetLoweringInfo.cpp - Encapsulate target details ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics the TargetCodeGenInfo class from the file
// clang/lib/CodeGen/TargetInfo.cpp.
//
//===----------------------------------------------------------------------===//

#include "TargetLoweringInfo.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace cir {

TargetLoweringInfo::~TargetLoweringInfo() = default;

cir::SyncScopeKind
TargetLoweringInfo::convertSyncScope(cir::SyncScopeKind syncScope) const {
  // By default, targets don't deal with sync scopes other than system scope.
  return cir::SyncScopeKind::System;
}

mlir::Value TargetLoweringInfo::lowerAggregateVAArg(
    CIRBaseBuilderTy &builder, cir::VAArgOp op, mlir::Value valist,
    const cir::CIRDataLayout &dataLayout) const {
  op.emitError() << "ClangIR code gen Not Yet Implemented: "
                 << "va_arg of an aggregate type on this target";
  return {};
}

} // namespace cir
