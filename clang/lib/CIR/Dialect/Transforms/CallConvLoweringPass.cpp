//===- CallConvLoweringPass.cpp - Lower CIR to ABI calling convention ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass walks every cir.func and cir.call in the module, computes a
// FunctionClassification for it (via either an ABI target or a pre-built
// classification injected as a function attribute), and dispatches to
// CIRABIRewriteContext to perform the actual IR rewriting.
//
// Two driver modes (mutually exclusive):
//
//   target=test
//     Use the MLIR test ABI target (mlir/lib/ABI/Targets/Test/) to classify
//     each function.  Predictable rules that approximate x86_64 SysV.  Real
//     targets (x86_64, AArch64) will be added once the LLVM ABI library
//     ships them.
//
//   classification-attr=<name>
//     Read a DictionaryAttr named <name> from each cir.func and parse it via
//     mlir::abi::test::parseClassificationAttr.  Used by tests to inject any
//     classification (including shapes the test target itself does not
//     produce) without depending on a real ABI target.
//
// The pass requires a `dlti.dl_spec` attribute on the module so the
// classifier can query type sizes and alignments.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "TargetLowering/CIRABIRewriteContext.h"

#include "mlir/ABI/ABIRewriteContext.h"
#include "mlir/ABI/Targets/Test/TestTarget.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace mlir::abi;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CALLCONVLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

bool needsRewrite(const FunctionClassification &fc) {
  if ((fc.returnInfo.kind != ArgKind::Direct) || fc.returnInfo.coercedType)
    return true;
  for (const ArgClassification &ac : fc.argInfos)
    if ((ac.kind != ArgKind::Direct) || ac.coercedType)
      return true;
  return false;
}

struct CallConvLoweringPass
    : public impl::CallConvLoweringBase<CallConvLoweringPass> {
  using CallConvLoweringBase::CallConvLoweringBase;
  void runOnOperation() override;
};

/// Classify \p func using whichever driver mode is configured.  Returns
/// std::nullopt and emits an error on the function if classification fails
/// (e.g. injection-driver mode but the function is missing the attribute,
/// or the attribute is malformed).
std::optional<FunctionClassification>
classifyFunction(cir::FuncOp func, const DataLayout &dl, StringRef target,
                 StringRef classificationAttrName) {
  ArrayRef<Type> argTypes = func.getFunctionType().getInputs();
  Type returnType = func.getFunctionType().getReturnType();

  if (!classificationAttrName.empty()) {
    auto attr = func->getAttrOfType<DictionaryAttr>(classificationAttrName);
    if (!attr) {
      func.emitOpError()
          << "missing classification attribute '" << classificationAttrName
          << "' (CallConvLowering driver mode 'classification-attr')";
      return std::nullopt;
    }
    return mlir::abi::test::parseClassificationAttr(
        attr, [&]() { return func.emitOpError(); });
  }

  if (target == "test")
    return mlir::abi::test::classify(argTypes, returnType, dl);

  func.emitOpError() << "unknown target '" << target << "' (supported: test)";
  return std::nullopt;
}

/// Find the cir.func declaration matching a direct cir.call / cir.try_call
/// callee, if any.  Returns nullptr if the callee is indirect or the symbol
/// cannot be resolved.  Takes a SymbolTable instead of a ModuleOp so the
/// symbol lookup is amortized across all the call sites the driver walks
/// (ModuleOp::lookupSymbol is linear per call).
cir::FuncOp lookupCallee(Operation *callOp, SymbolTable &symbolTable) {
  FlatSymbolRefAttr callee;
  if (auto call = dyn_cast<cir::CallOp>(callOp))
    callee = call.getCalleeAttr();
  else if (auto tryCall = dyn_cast<cir::TryCallOp>(callOp))
    callee = tryCall.getCalleeAttr();
  else
    return nullptr;
  if (!callee)
    return nullptr;
  return symbolTable.lookup<cir::FuncOp>(callee.getValue());
}

void CallConvLoweringPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  if (target.empty() == classificationAttr.empty()) {
    moduleOp.emitOpError() << "CallConvLowering requires exactly one of "
                              "'target' or 'classification-attr' pass options";
    signalPassFailure();
    return;
  }

  if (!moduleOp->hasAttr(DLTIDialect::kDataLayoutAttrName)) {
    moduleOp.emitOpError()
        << "CallConvLowering requires a DataLayout (dlti.dl_spec attribute "
           "on the module)";
    signalPassFailure();
    return;
  }

  DataLayout dl(moduleOp);
  CIRABIRewriteContext rewriteCtx(moduleOp);
  SymbolTable symbolTable(moduleOp);

  // Classify every cir.func up front.  No IR mutation happens here, so
  // later walks can consult any function's classification regardless of
  // visitation order.
  llvm::MapVector<cir::FuncOp, FunctionClassification> classifications;
  bool anyFailed = false;
  moduleOp.walk([&](cir::FuncOp f) {
    auto fc = classifyFunction(f, dl, target, classificationAttr);
    if (!fc) {
      anyFailed = true;
      return;
    }
    classifications.insert({f, std::move(*fc)});
  });
  if (anyFailed) {
    signalPassFailure();
    return;
  }

  // Build a callee-to-callers index.  One module walk collects every direct
  // cir.call / cir.try_call to each cir.func; the loop below rewrites a
  // function and all of its call sites together.  Indirect or unresolved
  // callees are skipped here; rewriteCallSite errors on those at the end.
  llvm::DenseMap<cir::FuncOp, SmallVector<Operation *>> callers;
  moduleOp.walk([&](Operation *op) {
    if (!isa<cir::CallOp, cir::TryCallOp>(op))
      return;
    if (cir::FuncOp callee = lookupCallee(op, symbolTable))
      callers[callee].push_back(op);
  });

  // Rewrite each function together with every direct call to it.  By the
  // time we move on to function F+1, F's signature and every direct call to
  // F have already been brought into alignment, and F+1..FN are still in
  // their original (mutually consistent) form, so the IR is verifier-clean
  // at every outer-iteration boundary.
  //
  // There is still a brief inner window where F's signature has been
  // rewritten but its callers have not yet caught up -- we have no way to
  // mutate both sides of a call atomically.  No verifier runs inside the
  // pass, and at pass exit the module is verifier-clean.  Fusing the inner
  // loop here keeps the invalid window per-function rather than module-wide.
  OpBuilder builder(ctx);
  for (auto &kv : classifications) {
    cir::FuncOp func = kv.first;
    const FunctionClassification &fc = kv.second;
    if (failed(rewriteCtx.rewriteFunctionDefinition(func, fc, builder))) {
      signalPassFailure();
      return;
    }
    for (Operation *callOp : callers.lookup(func)) {
      if (failed(rewriteCtx.rewriteCallSite(callOp, fc, builder))) {
        signalPassFailure();
        return;
      }
    }
  }

  // Reject indirect calls when the module contains any ABI rewrite that
  // would need call-site lowering.  We cannot strip or coerce operands
  // without a resolved callee symbol.
  const FunctionClassification *rewriteFc = nullptr;
  for (auto &kv : classifications) {
    if (needsRewrite(kv.second)) {
      rewriteFc = &kv.second;
      break;
    }
  }
  if (rewriteFc) {
    moduleOp.walk([&](cir::CallOp c) {
      if (!c.isIndirect())
        return;
      if (failed(rewriteCtx.rewriteCallSite(c, *rewriteFc, builder)))
        anyFailed = true;
    });
    if (anyFailed) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createCallConvLoweringPass() {
  return std::make_unique<CallConvLoweringPass>();
}
