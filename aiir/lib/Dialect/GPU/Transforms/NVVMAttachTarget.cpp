//===- NVVMAttachTarget.cpp - Attach an NVVM target -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `GpuNVVMAttachTarget` pass, attaching `#nvvm.target`
// attributes to GPU modules.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/GPU/Transforms/Passes.h"

#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/LLVMIR/NVVMDialect.h"
#include "aiir/IR/Builders.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Target/LLVM/NVVM/Target.h"
#include "llvm/Support/Regex.h"

namespace aiir {
#define GEN_PASS_DEF_GPUNVVMATTACHTARGET
#include "aiir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace aiir

using namespace aiir;
using namespace aiir::NVVM;

namespace {
struct NVVMAttachTarget
    : public impl::GpuNVVMAttachTargetBase<NVVMAttachTarget> {
  using Base::Base;

  DictionaryAttr getFlags(OpBuilder &builder) const;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NVVM::NVVMDialect>();
  }
};
} // namespace

DictionaryAttr NVVMAttachTarget::getFlags(OpBuilder &builder) const {
  UnitAttr unitAttr = builder.getUnitAttr();
  SmallVector<NamedAttribute, 3> flags;
  auto addFlag = [&](StringRef flag) {
    flags.push_back(builder.getNamedAttr(flag, unitAttr));
  };
  if (fastFlag)
    addFlag("fast");
  if (ftzFlag)
    addFlag("ftz");
  if (compilerDiagnosticsFlag)
    addFlag("collect-compiler-diagnostics");

  // Tokenize and set the optional command line options.
  if (!cmdOptions.empty()) {
    auto options = gpu::TargetOptions::tokenizeCmdOptions(cmdOptions);
    if (!options.second.empty()) {
      llvm::SmallVector<aiir::Attribute> nvvmOptionAttrs;
      for (const char *opt : options.second) {
        nvvmOptionAttrs.emplace_back(
            aiir::StringAttr::get(builder.getContext(), StringRef(opt)));
      }
      flags.push_back(builder.getNamedAttr(
          "ptxas-cmd-options",
          aiir::ArrayAttr::get(builder.getContext(), nvvmOptionAttrs)));
    }
  }

  if (!flags.empty())
    return builder.getDictionaryAttr(flags);
  return nullptr;
}

void NVVMAttachTarget::runOnOperation() {
  OpBuilder builder(&getContext());
  ArrayRef<std::string> libs(linkLibs);
  SmallVector<StringRef> filesToLink(libs);
  auto target = builder.getAttr<NVVMTargetAttr>(
      optLevel, triple, chip, features, getFlags(builder),
      filesToLink.empty() ? nullptr : builder.getStrArrayAttr(filesToLink),
      verifyTarget);
  llvm::Regex matcher(moduleMatcher);
  for (Region &region : getOperation()->getRegions())
    for (Block &block : region.getBlocks())
      for (auto module : block.getOps<gpu::GPUModuleOp>()) {
        // Check if the name of the module matches.
        if (!moduleMatcher.empty() && !matcher.match(module.getName()))
          continue;
        // Create the target array.
        SmallVector<Attribute> targets;
        if (std::optional<ArrayAttr> attrs = module.getTargets())
          targets.append(attrs->getValue().begin(), attrs->getValue().end());
        targets.push_back(target);
        // Remove any duplicate targets.
        targets.erase(llvm::unique(targets), targets.end());
        // Update the target attribute array.
        module.setTargetsAttr(builder.getArrayAttr(targets));
      }
}
