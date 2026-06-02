//====- LoweringPrepareCXXABI.h - Target ABI hooks for lowering prepare -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the target-ABI hooks that the cir-lowering-prepare pass
// needs while it rewrites target-dependent operations.  The pass itself stays
// target-agnostic and delegates the parts that depend on the target ABI to a
// per-target implementation selected by LoweringPrepareCXXABI::create.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_LOWERINGPREPARECXXABI_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_LOWERINGPREPARECXXABI_H

#include "mlir/Support/LLVM.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include <memory>

namespace clang {
class ASTContext;
} // namespace clang

namespace cir {

/// ABI-specific hooks used by the cir-lowering-prepare pass.  Mirrors the
/// split classic CodeGen draws between target-agnostic lowering and the
/// per-target ABI logic in `clang/lib/CodeGen/TargetInfo.cpp`.
class LoweringPrepareCXXABI {
public:
  /// Build the implementation for the target described by `astCtx`.  Targets
  /// without a specialized implementation get a base instance whose hooks
  /// report the requested lowering as not-yet-implemented.
  static std::unique_ptr<LoweringPrepareCXXABI>
  create(clang::ASTContext &astCtx);

  virtual ~LoweringPrepareCXXABI();

  /// Expand an aggregate `cir.va_arg` in place.  On success `op` has been
  /// replaced and erased.  On failure a not-yet-implemented diagnostic has
  /// been emitted on `op` and the caller must fail the pass.  Scalar
  /// `cir.va_arg` is lowered generically and never reaches this hook.
  ///
  /// The base implementation reports not-yet-implemented for every target;
  /// only targets with a specialized subclass expand the aggregate.
  virtual mlir::LogicalResult
  lowerAggregateVAArg(CIRBaseBuilderTy &builder, cir::VAArgOp op,
                      const cir::CIRDataLayout &datalayout);
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_LOWERINGPREPARECXXABI_H
