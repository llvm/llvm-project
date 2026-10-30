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
// `cir.offload.container` unit attribute), and splits such a module back into
// its per-target modules.
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
#include "clang/Basic/TargetID.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/InitAllDialects.h"
#include "clang/Driver/OffloadBundler.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <optional>
#include <string>

namespace {

llvm::cl::OptionCategory CIROffloadMergeCategory("cir-offload-merge options");

llvm::cl::opt<bool> Combine("combine", llvm::cl::desc("Combine CIR inputs"),
                            llvm::cl::cat(CIROffloadMergeCategory));

llvm::cl::opt<bool> Split("split", llvm::cl::desc("Split combined CIR input"),
                          llvm::cl::cat(CIROffloadMergeCategory));

llvm::cl::list<std::string>
    InputFileNames("input",
                   llvm::cl::desc("Input CIR file. Can be specified multiple "
                                  "times in combine mode."),
                   llvm::cl::cat(CIROffloadMergeCategory));

llvm::cl::list<std::string>
    TargetNames("targets", llvm::cl::CommaSeparated,
                llvm::cl::desc("[<offload kind>-<target triple>,...]"),
                llvm::cl::cat(CIROffloadMergeCategory));

llvm::cl::list<std::string>
    OutputFileNames("output",
                    llvm::cl::desc("Output CIR file. Can be specified "
                                   "multiple times in split mode."),
                    llvm::cl::cat(CIROffloadMergeCategory));

struct InputTarget {
  std::string Input;
  std::string Target;
  bool IsHost = false;
};

struct TargetOutput {
  std::string Target;
  std::string Output;
  bool Written = false;
};

int reportError(const llvm::Twine &message) {
  llvm::errs() << "error: " << message << '\n';
  return 1;
}

std::string sanitizeModuleName(llvm::StringRef target) {
  std::string name = "device_";
  for (char c : target) {
    if (std::isalnum(static_cast<unsigned char>(c)))
      name.push_back(c);
    else
      name.push_back('_');
  }
  return name;
}

std::string makeUnique(llvm::StringRef name,
                       llvm::StringMap<unsigned> &seenNames) {
  unsigned &count = seenNames[name];
  if (count++ == 0)
    return name.str();
  return (name + "_" + llvm::Twine(count - 1)).str();
}

bool isValidTarget(llvm::StringRef target,
                   const clang::OffloadBundlerConfig &bundlerConfig) {
  if (!clang::checkOffloadBundleID(target))
    return false;

  clang::OffloadTargetInfo offloadInfo(target, bundlerConfig);
  // The bundler has no "cuda" offload kind, but CUDA modules are combined here.
  return (offloadInfo.isOffloadKindValid() ||
          offloadInfo.OffloadKind == "cuda") &&
         offloadInfo.isTripleValid();
}

// Follows clang-offload-bundler's code object compatibility rules: the target
// ID of `bundleID` may leave out features that `target` specifies.
bool isCompatibleTarget(llvm::StringRef bundleID, llvm::StringRef target,
                        const clang::OffloadBundlerConfig &bundlerConfig) {
  clang::OffloadTargetInfo bundleInfo(bundleID, bundlerConfig);
  clang::OffloadTargetInfo targetInfo(target, bundlerConfig);
  if (bundleInfo == targetInfo)
    return true;

  if (!bundleInfo.isOffloadKindCompatible(targetInfo.OffloadKind) ||
      !bundleInfo.Triple.isCompatibleWith(targetInfo.Triple))
    return false;

  llvm::StringMap<bool> bundleFeatureMap;
  llvm::StringMap<bool> targetFeatureMap;
  std::optional<llvm::StringRef> bundleProcessor = clang::parseTargetID(
      bundleInfo.Triple, bundleInfo.TargetID, &bundleFeatureMap);
  std::optional<llvm::StringRef> targetProcessor = clang::parseTargetID(
      targetInfo.Triple, targetInfo.TargetID, &targetFeatureMap);
  if (!bundleProcessor || !targetProcessor ||
      *bundleProcessor != *targetProcessor)
    return false;

  for (const auto &bundleFeature : bundleFeatureMap) {
    auto targetFeature = targetFeatureMap.find(bundleFeature.getKey());
    if (targetFeature == targetFeatureMap.end() ||
        targetFeature->getValue() != bundleFeature.getValue())
      return false;
  }
  return true;
}

int validateCombineCommandLine(
    llvm::SmallVectorImpl<InputTarget> &inputTargets) {
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

  clang::OffloadBundlerConfig bundlerConfig;
  llvm::StringSet<> seenTargets;
  unsigned numHostTargets = 0;

  for (auto [input, target] : llvm::zip_equal(InputFileNames, TargetNames)) {
    if (!seenTargets.insert(target).second)
      return reportError("duplicate target '" + target + "'");
    if (!isValidTarget(target, bundlerConfig))
      return reportError("invalid target '" + target + "'");

    bool isHost = clang::OffloadTargetInfo(target, bundlerConfig).hasHostKind();
    numHostTargets += isHost;
    inputTargets.push_back({input, target, isHost});
  }

  if (numHostTargets != 1)
    return reportError("expected exactly one host target");
  if (inputTargets.size() == 1)
    return reportError("expected at least one device target");
  return 0;
}

int validateSplitCommandLine(
    llvm::SmallVectorImpl<TargetOutput> &targetOutputs) {
  if (InputFileNames.empty())
    return reportError("missing required -input");
  if (InputFileNames.size() != 1)
    return reportError("split mode expects exactly one input");
  if (TargetNames.empty())
    return reportError("missing required -targets");
  if (OutputFileNames.empty())
    return reportError("missing required -output");
  if (OutputFileNames.size() != TargetNames.size())
    return reportError("number of output files and targets should match in "
                       "split mode");

  clang::OffloadBundlerConfig bundlerConfig;
  llvm::StringSet<> seenTargets;
  llvm::StringSet<> seenOutputs;

  for (auto [target, output] : llvm::zip_equal(TargetNames, OutputFileNames)) {
    if (!seenTargets.insert(target).second)
      return reportError("duplicate target '" + target + "'");
    if (!seenOutputs.insert(output).second)
      return reportError("duplicate output '" + output + "'");
    if (!isValidTarget(target, bundlerConfig))
      return reportError("invalid target '" + target + "'");

    targetOutputs.push_back({target, output, false});
  }
  return 0;
}

void setOffloadAttrs(mlir::ModuleOp module, llvm::StringRef name,
                     cir::OffloadKind offloadKind, llvm::StringRef bundleID) {
  module.setSymName(name);
  module->setAttr(cir::CIRDialect::getOffloadKindAttrName(),
                  cir::OffloadKindAttr::get(module.getContext(), offloadKind));
  module->setAttr(cir::CIRDialect::getOffloadBundleIDAttrName(),
                  mlir::StringAttr::get(module.getContext(), bundleID));
}

mlir::OwningOpRef<mlir::ModuleOp>
combineInputs(llvm::ArrayRef<InputTarget> inputTargets,
              mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> hostModule;
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>> deviceModules;
  llvm::StringMap<unsigned> deviceNames;

  for (const InputTarget &inputTarget : inputTargets) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(inputTarget.Input,
                                              mlir::ParserConfig(&context));
    if (!module)
      return {};

    if (inputTarget.IsHost) {
      setOffloadAttrs(*module, "host", cir::OffloadKind::Host,
                      inputTarget.Target);
      hostModule = std::move(module);
      continue;
    }

    std::string deviceName =
        makeUnique(sanitizeModuleName(inputTarget.Target), deviceNames);
    setOffloadAttrs(*module, deviceName, cir::OffloadKind::Device,
                    inputTarget.Target);
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

int writeModuleToOutput(mlir::ModuleOp module, llvm::StringRef outputFileName) {
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

int splitInput(llvm::StringRef inputFileName,
               llvm::MutableArrayRef<TargetOutput> targetOutputs,
               mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> container =
      mlir::parseSourceFile<mlir::ModuleOp>(inputFileName,
                                            mlir::ParserConfig(&context));
  if (!container)
    return reportError("failed to parse input file");
  if (!(*container)->hasAttr(cir::CIRDialect::getOffloadContainerAttrName()))
    return reportError("expected input to be an offload container");

  clang::OffloadBundlerConfig bundlerConfig;

  // Written modules are detached from the container.
  for (mlir::ModuleOp module :
       llvm::make_early_inc_range(container->getOps<mlir::ModuleOp>())) {
    auto bundleIDAttr = module->getAttrOfType<mlir::StringAttr>(
        cir::CIRDialect::getOffloadBundleIDAttrName());
    if (!bundleIDAttr)
      return reportError("nested module is missing '" +
                         cir::CIRDialect::getOffloadBundleIDAttrName() + "'");

    llvm::StringRef bundleID = bundleIDAttr.getValue();
    if (!isValidTarget(bundleID, bundlerConfig))
      return reportError("invalid module bundle ID '" + bundleID + "'");

    TargetOutput *match = nullptr;
    for (TargetOutput &targetOutput : targetOutputs) {
      if (!isCompatibleTarget(bundleID, targetOutput.Target, bundlerConfig))
        continue;
      if (match)
        return reportError("module bundle ID '" + bundleID +
                           "' matches multiple requested targets");
      match = &targetOutput;
    }

    if (!match)
      continue;
    if (match->Written)
      return reportError("multiple modules match target '" + match->Target +
                         "'");

    module->remove();
    mlir::OwningOpRef<mlir::ModuleOp> detachedModule(module);
    if (int errorCode = writeModuleToOutput(*detachedModule, match->Output))
      return errorCode;
    match->Written = true;
  }

  for (const TargetOutput &targetOutput : targetOutputs)
    if (!targetOutput.Written)
      return reportError("can't find module for target '" +
                         targetOutput.Target + "'");
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::HideUnrelatedOptions(CIROffloadMergeCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "CIR host-device offload merge\n");

  if (Combine == Split)
    return reportError("expected exactly one of -combine or -split");

  mlir::DialectRegistry registry;
  cir::registerAllDialects(registry);
  registry.insert<mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect>();
  mlir::MLIRContext context(registry);
  context.loadDialect<cir::CIRDialect>();

  if (Split) {
    llvm::SmallVector<TargetOutput> targetOutputs;
    if (int errorCode = validateSplitCommandLine(targetOutputs))
      return errorCode;
    return splitInput(InputFileNames.front(), targetOutputs, context);
  }

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
