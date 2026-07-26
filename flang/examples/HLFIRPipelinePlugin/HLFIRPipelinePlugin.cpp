//===-- HLFIRPipelinePlugin.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Example plugin adding an out-of-tree MLIR pass to flang's HLFIR-to-FIR pass
// pipeline, at the points where the HLFIR intrinsic operations (hlfir.sum,
// hlfir.matmul, ...) are still present. The pass prints those operations,
// tagged with the pipeline position it was inserted at.
//
// It is exposed through both plugin entry points. For `flang -fc1 -load`, a
// static initializer calls fir::registerPassPipelineConfigCallback and hooks
// the pass onto the HLFIROptEarly and HLFIROptLast extension points. For
// fir-opt, mlirGetPassPluginInfo makes it available to --load-pass-plugin:
//
//   fir-opt --load-pass-plugin=./flangHLFIRPipelinePlugin.so \
//           --pass-pipeline='builtin.module(print-hlfir-intrinsics)'
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Passes/Pipelines.h"
#include "flang/Tools/CrossToolHelpers.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

namespace {

/// Print every HLFIR operation still present in the module, tagged with a
/// caller-supplied label. Matches on the `hlfir` dialect namespace rather than
/// a hard-coded op list, so the example does not need to link the HLFIR dialect
/// library.
struct PrintHLFIRIntrinsicsPass
    : public mlir::PassWrapper<PrintHLFIRIntrinsicsPass,
          mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintHLFIRIntrinsicsPass)

  PrintHLFIRIntrinsicsPass() = default;
  explicit PrintHLFIRIntrinsicsPass(llvm::StringRef labelValue) {
    label = labelValue.str();
  }
  PrintHLFIRIntrinsicsPass(const PrintHLFIRIntrinsicsPass &other)
      : mlir::PassWrapper<PrintHLFIRIntrinsicsPass,
            mlir::OperationPass<mlir::ModuleOp>>(other) {
    label = other.label;
  }

  llvm::StringRef getArgument() const override {
    return "print-hlfir-intrinsics";
  }
  llvm::StringRef getDescription() const override {
    return "Print the HLFIR operations that are still present in the module";
  }

  Option<std::string> label{*this, "label",
      llvm::cl::desc("Tag prefixed to every printed line, identifying the "
                     "pipeline position this pass was inserted at"),
      llvm::cl::init("hlfir")};

  void runOnOperation() override {
    llvm::outs() << "[" << label << "] begin\n";
    getOperation().walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() == "hlfir") {
        llvm::outs() << "[" << label << "] " << op->getName().getStringRef()
                     << "\n";
      }
    });
    llvm::outs() << "[" << label << "] end\n";
  }
};

struct FlangPipelineRegistration {
  FlangPipelineRegistration() {
    fir::registerPassPipelineConfigCallback(
        [](MLIRToLLVMPassPipelineConfig &config) {
          config.registerHLFIROptEarlyEPCallbacks(
              [](mlir::PassManager &pm, llvm::OptimizationLevel) {
                pm.addPass(
                    std::make_unique<PrintHLFIRIntrinsicsPass>("hlfir-early"));
              });
          config.registerHLFIROptLastEPCallbacks(
              [](mlir::PassManager &pm, llvm::OptimizationLevel) {
                pm.addPass(
                    std::make_unique<PrintHLFIRIntrinsicsPass>("hlfir-last"));
              });
        });
  }
};

// The constructor runs when the shared object is loaded, before any compilation
// starts.
static FlangPipelineRegistration flangPipelineRegistration;

} // namespace

/// Entry point used by fir-opt's --load-pass-plugin.
extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "HLFIRPipelinePlugin", LLVM_VERSION_STRING,
      []() { mlir::PassRegistration<PrintHLFIRIntrinsicsPass>(); }};
}
