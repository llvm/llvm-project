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
// This file handles Direct (pass-through and coerce-in-registers), Extend,
// Ignore, Indirect (sret return, byval and non-byval arguments), and Expand
// (struct flattening into scalar fields).
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRABIREWRITECONTEXT_H
#define CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRABIREWRITECONTEXT_H

#include "mlir/ABI/ABIRewriteContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <utility>

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

  ~CIRABIRewriteContext() {
    assert(pendingParamSlots.empty() &&
           "finalizeParameterSlots must run before the rewrite context dies");
  }

  mlir::LogicalResult
  rewriteFunctionDefinition(mlir::FunctionOpInterface funcOp,
                            const mlir::abi::FunctionClassification &fc,
                            mlir::OpBuilder &builder) override;

  mlir::LogicalResult
  rewriteCallSite(mlir::Operation *callOp,
                  const mlir::abi::FunctionClassification &fc,
                  mlir::OpBuilder &builder) override;

  /// Expand a `cir.va_arg` into the x86-64 SysV register-save-area /
  /// overflow-area sequence.
  mlir::LogicalResult rewriteVAArg(mlir::Operation *vaArgOp,
                                   const mlir::abi::ArgClassification &ac,
                                   mlir::OpBuilder &builder) override;

  /// Retype \p addrOp, which holds the address of \p funcOp, to the signature
  /// funcOp was rewritten to, and cast it back so existing uses keep the type
  /// they were built for.  A no-op when the ABI left funcOp's type alone.
  /// Call after funcOp has been rewritten.  Not an override, since taking a
  /// function's address has no counterpart in the generic contract.
  void rewriteFunctionAddress(cir::GetGlobalOp addrOp, cir::FuncOp funcOp,
                              mlir::OpBuilder &builder);

  /// Bring each non-byval indirect parameter of \p funcOp into the shape the
  /// rest of the rewrite assumes: the parameter's only use, if it has one, is
  /// a single store into an alloca of the matching pointer type that no other
  /// non-byval indirect parameter spills to, and that alloca states the
  /// alignment the ABI promises rather than the one CIRGen picked for a local
  /// copy.  A use of the parameter as a call argument is routed through a load
  /// of that slot, so that it names the storage an argument has to name.
  ///
  /// Call for every function before any definition or call site is rewritten.
  /// findParamSpill asserts this shape while the enclosing definition is
  /// rewritten, and a call is rewritten with its callee rather than with its
  /// enclosing function and so may be reached first.
  ///
  /// A parameter read with no spill to name is given one, since it becomes
  /// the incoming pointer directly.  A parameter spilled twice, spilled to a
  /// slot another such parameter also spills to, consumed other than as a
  /// call argument, spilled where the incoming pointer cannot replace the
  /// storage, or read where the spill does not dominate it gets a diagnostic
  /// on the operation at fault and failure.
  ///
  /// Not an override, since this has no counterpart in the generic contract.
  mlir::LogicalResult
  prepareNonByvalParameters(cir::FuncOp funcOp,
                            const mlir::abi::FunctionClassification &fc);

  /// Replace each non-byval indirect parameter's spill slot with the
  /// incoming pointer, so the body operates on the caller's storage in place.
  /// Call once, after every function and call site has been rewritten: a call
  /// forwarding such a parameter reads the slot to recognize it.  Not an
  /// override, since deferring this has no counterpart in the generic
  /// contract.
  void finalizeParameterSlots();

  mlir::StringRef getDialectNamespace() const override { return "cir"; }

private:
  mlir::ModuleOp module;
  const mlir::DataLayout &dl;

  /// Param-slot allocas that non-byval indirect parameters will
  /// replace, paired with the incoming pointer that replaces them.  The
  /// rewrite retypes the block argument but leaves the slot standing, because
  /// a call site recognizes a forwardable parameter by the slot its operand
  /// was loaded from.  finalizeParameterSlots does the replacement once every
  /// call site has been rewritten.
  llvm::SmallVector<std::pair<cir::AllocaOp, mlir::BlockArgument>>
      pendingParamSlots;
};

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRABIREWRITECONTEXT_H
