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

/// Find the cir.func declaration matching a cir.call's callee, if any.
/// Returns nullptr if the callee is indirect or the symbol cannot be
/// resolved (in which case the call is left alone).
cir::FuncOp lookupCallee(cir::CallOp call, ModuleOp module) {
  FlatSymbolRefAttr callee = call.getCalleeAttr();
  if (!callee)
    return nullptr;
  return module.lookupSymbol<cir::FuncOp>(callee.getValue());
}

void CallConvLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = &getContext();

  if (target.empty() == classificationAttr.empty()) {
    module.emitOpError() << "CallConvLowering requires exactly one of "
                            "'target' or 'classification-attr' pass options";
    signalPassFailure();
    return;
  }

  if (!module->hasAttr(DLTIDialect::kDataLayoutAttrName)) {
    module.emitOpError()
        << "CallConvLowering requires a DataLayout (dlti.dl_spec attribute "
           "on the module)";
    signalPassFailure();
    return;
  }

  DataLayout dl(module);
  CIRABIRewriteContext rewriteCtx(module);

  // Pre-compute classifications for every cir.func so that call-site
  // rewriting can find them (call site uses callee's classification).
  llvm::MapVector<cir::FuncOp, FunctionClassification> classifications;
  bool anyFailed = false;
  module.walk([&](cir::FuncOp f) {
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

  OpBuilder rewriter(ctx);

  // Rewrite call sites first, while functions still have their original
  // signatures.  This avoids any chance of us reading a partially-rewritten
  // signature and matching args against the wrong classification.
  SmallVector<cir::CallOp> calls;
  module.walk([&](cir::CallOp c) { calls.push_back(c); });
  for (cir::CallOp call : calls) {
    cir::FuncOp callee = lookupCallee(call, module);
    if (!callee)
      continue;
    auto it = classifications.find(callee);
    if (it == classifications.end())
      continue;
    if (failed(rewriteCtx.rewriteCallSite(call, it->second, rewriter))) {
      signalPassFailure();
      return;
    }
  }

  // Now rewrite each function definition.
  for (auto &kv : classifications) {
    if (failed(rewriteCtx.rewriteFunctionDefinition(kv.first, kv.second,
                                                    rewriter))) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createCallConvLoweringPass() {
  return std::make_unique<CallConvLoweringPass>();
}
