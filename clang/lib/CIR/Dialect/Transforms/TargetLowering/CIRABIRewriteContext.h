//===- CIRABIRewriteContext.h - CIR-specific ABI rewriting ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines CIRABIRewriteContext, the CIR dialect's implementation of
// the shared ABIRewriteContext interface.  It rewrites CIR function definitions
// and call sites to match ABI-lowered signatures.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRABIREWRITECONTEXT_H
#define CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRABIREWRITECONTEXT_H

#include "mlir/ABI/ABIRewriteContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace cir {

/// CIR-specific implementation of the ABIRewriteContext interface.
///
/// This class knows how to rewrite CIR FuncOps and CallOps to match
/// ABI-lowered signatures, using CIR operations for coercion (alloca,
/// load, store, cast, etc.).
class CIRABIRewriteContext : public mlir::abi::ABIRewriteContext {
  mlir::ModuleOp module;
  bool passByValueIsNoAlias = false;

public:
  explicit CIRABIRewriteContext(mlir::ModuleOp module,
                                bool passByValueIsNoAlias = false)
      : module(module), passByValueIsNoAlias(passByValueIsNoAlias) {}

  mlir::LogicalResult
  rewriteFunctionDefinition(mlir::FunctionOpInterface funcOp,
                            const mlir::abi::FunctionClassification &fc,
                            mlir::OpBuilder &rewriter) override;

  mlir::LogicalResult
  rewriteCallSite(mlir::Operation *callOp,
                  const mlir::abi::FunctionClassification &fc,
                  mlir::OpBuilder &rewriter) override;

  mlir::StringRef getDialectNamespace() const override { return "cir"; }
};

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRABIREWRITECONTEXT_H
