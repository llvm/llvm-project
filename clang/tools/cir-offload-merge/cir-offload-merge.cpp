//===-- cir-offload-merge/cir-offload-merge.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This tool combines a host CIR module and its device CIR modules into a
// single module marked as an offload container (carrying the
// `cir.offload.container` unit attribute).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/InitAllDialects.h"
#include "clang/Driver/OffloadBundler.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

static llvm::cl::OptionCategory
    CIROffloadMergeCategory("cir-offload-merge options");

static llvm::cl::opt<bool> Combine("combine",
                                   llvm::cl::desc("Combine CIR inputs"),
                                   llvm::cl::cat(CIROffloadMergeCategory));

static llvm::cl::list<std::string>
    InputFileNames("input",
                   llvm::cl::desc("Input CIR file. Can be specified multiple "
                                  "times."),
                   llvm::cl::cat(CIROffloadMergeCategory));

static llvm::cl::list<std::string>
    TargetNames("targets", llvm::cl::CommaSeparated,
                llvm::cl::desc("[<offload kind>-<target triple>,...]"),
                llvm::cl::cat(CIROffloadMergeCategory));

static llvm::cl::list<std::string>
    OutputFileNames("output", llvm::cl::desc("Output CIR file."),
                    llvm::cl::cat(CIROffloadMergeCategory));

namespace {
struct InputTarget {
  std::string Input;
  bool IsHost = false;
};
} // namespace

static int reportError(const llvm::Twine &message) {
  llvm::errs() << "error: " << message << '\n';
  return 1;
}

// Targets are offload bundle IDs, so they are checked with
// clang-offload-bundler's parser (thus, legalized). checkOffloadBundleID must
// run first, since OffloadTargetInfo asserts on a malformed ID.
static bool
isValidCombineTarget(llvm::StringRef target,
                     const clang::OffloadBundlerConfig &bundlerConfig) {
  if (!clang::checkOffloadBundleID(target))
    return false;

  clang::OffloadTargetInfo offloadInfo(target, bundlerConfig);
  // The bundler has no "cuda" offload kind, but CUDA modules are combined here.
  return (offloadInfo.isOffloadKindValid() ||
          offloadInfo.OffloadKind == "cuda") &&
         offloadInfo.isTripleValid();
}

static int
validateCombineCommandLine(llvm::SmallVectorImpl<InputTarget> &inputTargets) {
  if (InputFileNames.empty())
    return reportError("missing required -input");
  if (TargetNames.empty())
    return reportError("missing required -targets");
  if (OutputFileNames.empty())
    return reportError("missing required -output");
  if (OutputFileNames.size() != 1)
    return reportError("combine mode expects exactly one output");
  if (InputFileNames.size() != TargetNames.size())
    return reportError("number of input files and targets should match in "
                       "combine mode");

  // OffloadTargetInfo requires a bundler config; the defaults are enough to
  // parse a target.
  clang::OffloadBundlerConfig bundlerConfig;
  llvm::StringSet<> seenTargets;
  unsigned numHostTargets = 0;

  for (auto [input, target] : llvm::zip_equal(InputFileNames, TargetNames)) {
    if (!seenTargets.insert(target).second)
      return reportError("duplicate target '" + target + "'");
    if (!isValidCombineTarget(target, bundlerConfig))
      return reportError("invalid target '" + target + "'");

    bool isHost = clang::OffloadTargetInfo(target, bundlerConfig).hasHostKind();
    numHostTargets += isHost;
    inputTargets.push_back({input, isHost});
  }

  if (numHostTargets != 1)
    return reportError("expected exactly one host target");
  if (inputTargets.size() == 1)
    return reportError("expected at least one device target");
  return 0;
}

static void setOffloadAttrs(mlir::ModuleOp module, llvm::StringRef name,
                            cir::OffloadKind offloadKind) {
  module.setSymName(name);
  module->setAttr(cir::CIRDialect::getOffloadKindAttrName(),
                  cir::OffloadKindAttr::get(module.getContext(), offloadKind));
}

static mlir::OwningOpRef<mlir::ModuleOp>
combineInputs(llvm::ArrayRef<InputTarget> inputTargets,
              mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> hostModule;
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>> deviceModules;

  for (const InputTarget &inputTarget : inputTargets) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(inputTarget.Input,
                                              mlir::ParserConfig(&context));
    if (!module)
      return {};

    if (inputTarget.IsHost) {
      setOffloadAttrs(*module, "host", cir::OffloadKind::Host);
      hostModule = std::move(module);
      continue;
    }

    std::string deviceName =
        ("device_" + llvm::Twine(deviceModules.size())).str();
    setOffloadAttrs(*module, deviceName, cir::OffloadKind::Device);
    deviceModules.push_back(std::move(module));
  }

  // The container verifier expects the host module first, followed by the
  // device modules.
  mlir::ModuleOp container =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  container->setAttr(cir::CIRDialect::getOffloadContainerAttrName(),
                     mlir::UnitAttr::get(&context));
  container.getBody()->push_back(hostModule.release());
  for (mlir::OwningOpRef<mlir::ModuleOp> &deviceModule : deviceModules)
    container.getBody()->push_back(deviceModule.release());
  return container;
}

static int writeModuleToOutput(mlir::ModuleOp module,
                               llvm::StringRef outputFileName) {
  // Type aliases are only printed for an operation without a parent.
  assert(!module->getParentOp() && "expected a top-level module");
  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      mlir::openOutputFile(outputFileName, &errorMessage);
  if (!outputFile)
    return reportError(errorMessage);

  // TODO: Support bytecode output.
  module->print(outputFile->os());
  outputFile->os() << '\n';
  outputFile->keep();
  return 0;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::HideUnrelatedOptions(CIROffloadMergeCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "CIR host-device offload merge\n");

  if (!Combine)
    return reportError("expected -combine");

  mlir::DialectRegistry registry;
  cir::registerAllDialects(registry);
  registry.insert<mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect>();
  mlir::MLIRContext context(registry);
  context.loadDialect<cir::CIRDialect>();

  llvm::SmallVector<InputTarget> inputTargets;
  if (int errorCode = validateCombineCommandLine(inputTargets))
    return errorCode;

  mlir::OwningOpRef<mlir::ModuleOp> container =
      combineInputs(inputTargets, context);
  if (!container)
    return reportError("failed to parse input file");
  if (mlir::failed(mlir::verify(*container)))
    return reportError("failed to verify combined module");

  return writeModuleToOutput(*container, OutputFileNames.front());
}
