//===- CIRABIRewriteContext.h - CIR ABI rewrite context ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines CIRABIRewriteContext, the CIR dialect's implementation of the
// generic mlir::abi::ABIRewriteContext.  Given a FunctionClassification it
// rewrites a cir.func signature, the function body, and call sites to match
// the ABI-lowered shape.
//
// This file currently handles Direct (pass-through and coerce-in-registers),
// Extend, and Ignore.  The remaining ArgKind handlers (Indirect, Expand)
// are added by subsequent PRs in the calling-convention-lowering split
// series.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRABIREWRITECONTEXT_H
#define CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRABIREWRITECONTEXT_H

#include "mlir/ABI/ABIRewriteContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace cir {

/// CIR-specific implementation of mlir::abi::ABIRewriteContext.
///
/// The driver pass (CallConvLoweringPass) computes a FunctionClassification
/// for each cir.func / cir.call and dispatches to this class to perform the
/// actual IR rewriting using cir dialect operations.
///
/// Holds a reference to the module's DataLayout for coercion alignment
/// queries.  The DataLayout must outlive the rewrite context.
class CIRABIRewriteContext : public mlir::abi::ABIRewriteContext {
public:
  CIRABIRewriteContext(mlir::ModuleOp module, const mlir::DataLayout &dl)
      : module(module), dl(dl) {}

  mlir::LogicalResult
  rewriteFunctionDefinition(mlir::FunctionOpInterface funcOp,
                            const mlir::abi::FunctionClassification &fc,
                            mlir::OpBuilder &builder) override;

  mlir::LogicalResult
  rewriteCallSite(mlir::Operation *callOp,
                  const mlir::abi::FunctionClassification &fc,
                  mlir::OpBuilder &builder) override;

  mlir::StringRef getDialectNamespace() const override { return "cir"; }

private:
  mlir::ModuleOp module;
  const mlir::DataLayout &dl;
};

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRABIREWRITECONTEXT_H
