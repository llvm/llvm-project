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
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/Analysis/CIRAliasAnalysis.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <memory>
#include <utility>

namespace cir {

class CIRBaseBuilderTy;

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
  CIRABIRewriteContext(mlir::ModuleOp module, const mlir::DataLayout &dl,
                       bool enableNoundefAttrs)
      : module(module), dl(dl), cirDataLayout(module),
        enableNoundefAttrs(enableNoundefAttrs), aliasAnalysis(module) {
    cir::registerCIRAliasAnalyses(aliasAnalysis, module);
  }

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

  /// Replace a `cir.va_arg` of an ignored type with poison, leaving the
  /// va_list unchanged.  Every other `cir.va_arg` is passed to expandVAArg.
  mlir::LogicalResult rewriteVAArg(mlir::Operation *vaArgOp,
                                   const mlir::abi::ArgClassification &ac,
                                   mlir::OpBuilder &builder) final;

  /// Retype \p addrOp, which holds the address of \p funcOp, to the signature
  /// funcOp was rewritten to, and cast it back so existing uses keep the type
  /// they were built for.  A no-op when the ABI left funcOp's type alone.
  /// Call after funcOp has been rewritten.  Not an override, since taking a
  /// function's address has no counterpart in the generic contract.
  void rewriteFunctionAddress(cir::GetGlobalOp addrOp, cir::FuncOp funcOp,
                              mlir::OpBuilder &builder);

  /// Bring each non-byval indirect parameter of \p funcOp into the shape the
  /// rest of the rewrite assumes: the parameter's only use, if it has one, is
  /// a single store into an alloca of the incoming pointer's type, directly or
  /// through an address_space cast of it.  No other non-byval indirect
  /// parameter spills to that alloca, and it states the alignment the ABI
  /// promises rather than the one CIRGen picked for a local copy.  A use of
  /// the parameter as a call argument is routed through a load of that slot,
  /// so that it names the storage an argument has to name.
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

protected:
  /// Expand \p op, whose type is not ignored, into a read through the
  /// target's va_list that also advances the va_list past the value.
  virtual mlir::LogicalResult
  expandVAArg(cir::VAArgOp op, const mlir::abi::ArgClassification &ac,
              CIRBaseBuilderTy &builder);

  mlir::ModuleOp module;
  const mlir::DataLayout &dl;

  /// The CIR view of the module's data layout, which provides the store and
  /// alloc sizes that \c dl does not.
  cir::CIRDataLayout cirDataLayout;

private:
  /// Whether extended and indirect arguments are marked llvm.noundef.  Set
  /// from the enable_noundef_attrs field of the module's
  /// #cir.lowering_options.
  bool enableNoundefAttrs;

  /// Alias analysis over the module with the CIR implementations registered.
  /// rewriteCallSite asks it whether a byval operand's storage is written
  /// between its load and the call.
  mlir::AliasAnalysis aliasAnalysis;

  /// Param-slot allocas that non-byval indirect parameters will
  /// replace, paired with the incoming pointer that replaces them.  The
  /// rewrite retypes the block argument but leaves the slot standing, because
  /// a call site recognizes a forwardable parameter by the slot its operand
  /// was loaded from.  finalizeParameterSlots does the replacement once every
  /// call site has been rewritten.
  llvm::SmallVector<std::pair<cir::AllocaOp, mlir::BlockArgument>>
      pendingParamSlots;
};

/// The signature the indirect call \p call reaches its callee through.
cir::FuncType getIndirectCalleeType(cir::CIRCallOpInterface call);

/// A rewrite context that expands `cir.va_arg` through the x86-64 SysV
/// va_list.
std::unique_ptr<CIRABIRewriteContext>
createX86_64ABIRewriteContext(mlir::ModuleOp module, const mlir::DataLayout &dl,
                              bool enableNoundefAttrs);

/// A rewrite context that expands `cir.va_arg` through the AMDGPU pointer
/// va_list.
std::unique_ptr<CIRABIRewriteContext>
createAMDGPUABIRewriteContext(mlir::ModuleOp module, const mlir::DataLayout &dl,
                              bool enableNoundefAttrs);

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRABIREWRITECONTEXT_H
