//===-- XeVMAttachTarget.cpp - Attach an XeVM target ----------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `GpuXeVMAttachTarget` pass, attaching `#xevm.target`
// attributes to GPU modules.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/GPU/Transforms/Passes.h"

#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/LLVMIR/XeVMDialect.h"
#include "aiir/IR/Builders.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Target/LLVM/XeVM/Target.h"
#include "llvm/Support/Regex.h"

namespace aiir {
#define GEN_PASS_DEF_GPUXEVMATTACHTARGET
#include "aiir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace aiir

using namespace aiir;
using namespace aiir::xevm;

namespace {
struct XeVMAttachTarget
    : public aiir::impl::GpuXeVMAttachTargetBase<XeVMAttachTarget> {
  using Base::Base;

  DictionaryAttr getFlags(OpBuilder &builder) const;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<xevm::XeVMDialect>();
  }
};
} // namespace

DictionaryAttr XeVMAttachTarget::getFlags(OpBuilder &builder) const {
  SmallVector<NamedAttribute, 3> flags;
  // Tokenize and set the optional command line options.
  if (!cmdOptions.empty()) {
    std::pair<llvm::BumpPtrAllocator, SmallVector<const char *>> options =
        gpu::TargetOptions::tokenizeCmdOptions(cmdOptions);
    if (!options.second.empty()) {
      llvm::SmallVector<aiir::Attribute> xevmOptionAttrs;
      for (const char *opt : options.second) {
        xevmOptionAttrs.emplace_back(
            aiir::StringAttr::get(builder.getContext(), StringRef(opt)));
      }
      flags.push_back(builder.getNamedAttr(
          "cmd-options",
          aiir::ArrayAttr::get(builder.getContext(), xevmOptionAttrs)));
    }
  }

  if (!flags.empty())
    return builder.getDictionaryAttr(flags);
  return nullptr;
}

void XeVMAttachTarget::runOnOperation() {
  OpBuilder builder(&getContext());
  ArrayRef<std::string> libs(linkLibs);
  SmallVector<StringRef> filesToLink(libs);
  auto target = builder.getAttr<xevm::XeVMTargetAttr>(
      optLevel, triple, chip, getFlags(builder),
      filesToLink.empty() ? nullptr : builder.getStrArrayAttr(filesToLink));
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
