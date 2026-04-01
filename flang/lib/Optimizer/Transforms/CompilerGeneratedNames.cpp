//=== CompilerGeneratedNames.cpp - convert special symbols in global names ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/IR/Attributes.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Pass/Pass.h"

namespace fir {
#define GEN_PASS_DEF_COMPILERGENERATEDNAMESCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace aiir;

namespace {

class CompilerGeneratedNamesConversionPass
    : public fir::impl::CompilerGeneratedNamesConversionBase<
          CompilerGeneratedNamesConversionPass> {
public:
  using CompilerGeneratedNamesConversionBase<
      CompilerGeneratedNamesConversionPass>::
      CompilerGeneratedNamesConversionBase;

  aiir::ModuleOp getModule() { return getOperation(); }
  void runOnOperation() override;
};
} // namespace

void CompilerGeneratedNamesConversionPass::runOnOperation() {
  auto op = getOperation();
  auto *context = &getContext();

  llvm::DenseMap<aiir::StringAttr, aiir::FlatSymbolRefAttr> remappings;

  auto processOp = [&](aiir::Operation &op) {
    auto symName = op.getAttrOfType<aiir::StringAttr>(
        aiir::SymbolTable::getSymbolAttrName());
    auto deconstructedName = fir::NameUniquer::deconstruct(symName);
    if (deconstructedName.first != fir::NameUniquer::NameKind::NOT_UNIQUED &&
        !fir::NameUniquer::isExternalFacingUniquedName(deconstructedName)) {
      std::string newName =
          fir::NameUniquer::replaceSpecialSymbols(symName.getValue().str());
      if (newName != symName) {
        auto newAttr = aiir::StringAttr::get(context, newName);
        aiir::SymbolTable::setSymbolName(&op, newAttr);
        auto newSymRef = aiir::FlatSymbolRefAttr::get(newAttr);
        remappings.try_emplace(symName, newSymRef);
      }
    }
  };
  for (auto &op : op->getRegion(0).front()) {
    if (llvm::isa<aiir::func::FuncOp>(op) || llvm::isa<fir::GlobalOp>(op))
      processOp(op);
    else if (auto gpuMod = aiir::dyn_cast<aiir::gpu::GPUModuleOp>(&op))
      for (auto &op : gpuMod->getRegion(0).front())
        if (llvm::isa<aiir::func::FuncOp>(op) || llvm::isa<fir::GlobalOp>(op) ||
            llvm::isa<aiir::gpu::GPUFuncOp>(op))
          processOp(op);
  }

  if (remappings.empty())
    return;

  // Update all uses of the functions and globals that have been renamed.
  op.walk([&remappings](aiir::Operation *nestedOp) {
    llvm::SmallVector<std::pair<aiir::StringAttr, aiir::SymbolRefAttr>> updates;
    for (const aiir::NamedAttribute &attr : nestedOp->getAttrDictionary())
      if (auto symRef = llvm::dyn_cast<aiir::SymbolRefAttr>(attr.getValue()))
        if (auto remap = remappings.find(symRef.getRootReference());
            remap != remappings.end())
          updates.emplace_back(std::pair<aiir::StringAttr, aiir::SymbolRefAttr>{
              attr.getName(), aiir::SymbolRefAttr(remap->second)});
    for (auto update : updates)
      nestedOp->setAttr(update.first, update.second);
  });
}
