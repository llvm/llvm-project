//===- ExternalNameConversion.cpp -- convert name with external convention ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace fir {
#define GEN_PASS_DEF_EXTERNALNAMECONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Mangle the name with gfortran convention.
std::string
mangleExternalName(const std::pair<fir::NameUniquer::NameKind,
                                   fir::NameUniquer::DeconstructedName>
                       result,
                   bool appendUnderscore) {
  if (result.first == fir::NameUniquer::NameKind::COMMON &&
      result.second.name.empty())
    return Fortran::common::blankCommonObjectName;
  return Fortran::common::GetExternalAssemblyName(result.second.name,
                                                  appendUnderscore);
}

namespace {

class ExternalNameConversionPass
    : public fir::impl::ExternalNameConversionBase<ExternalNameConversionPass> {
public:
  using ExternalNameConversionBase<
      ExternalNameConversionPass>::ExternalNameConversionBase;

  mlir::ModuleOp getModule() { return getOperation(); }
  void runOnOperation() override;
};
} // namespace

void ExternalNameConversionPass::runOnOperation() {
  auto op = getOperation();
  auto *context = &getContext();

  llvm::DenseMap<mlir::StringAttr, mlir::FlatSymbolRefAttr> remappings;

  auto processFctOrGlobal = [&](mlir::Operation &funcOrGlobal) {
    auto symName = funcOrGlobal.getAttrOfType<mlir::StringAttr>(
        mlir::SymbolTable::getSymbolAttrName());
    auto deconstructedName = fir::NameUniquer::deconstruct(symName);
    if (fir::NameUniquer::isExternalFacingUniquedName(deconstructedName)) {
      auto newName = mangleExternalName(deconstructedName, appendUnderscoreOpt);
      auto newAttr = mlir::StringAttr::get(context, newName);
      mlir::SymbolTable::setSymbolName(&funcOrGlobal, newAttr);
      auto newSymRef = mlir::FlatSymbolRefAttr::get(newAttr);
      remappings.try_emplace(symName, newSymRef);
      if (llvm::isa<mlir::func::FuncOp>(funcOrGlobal))
        funcOrGlobal.setAttr(fir::getInternalFuncNameAttrName(), symName);
    }
  };

  auto renameFuncOrGlobalInModule = [&](mlir::Operation *module) {
    for (auto &op : module->getRegion(0).front()) {
      if (mlir::isa<mlir::func::FuncOp, fir::GlobalOp>(op)) {
        processFctOrGlobal(op);
      } else if (auto gpuMod = mlir::dyn_cast<mlir::gpu::GPUModuleOp>(op)) {
        for (auto &gpuOp : gpuMod.getBodyRegion().front())
          if (mlir::isa<mlir::func::FuncOp, fir::GlobalOp,
                        mlir::gpu::GPUFuncOp>(gpuOp))
            processFctOrGlobal(gpuOp);
      }
    }
  };

  // Update names of external Fortran functions and names of Common Block
  // globals.
  renameFuncOrGlobalInModule(op);

  if (remappings.empty())
    return;

  // Update all uses of the functions and globals that have been renamed.
  op.walk([&remappings](mlir::Operation *nestedOp) {
    llvm::SmallVector<std::pair<mlir::StringAttr, mlir::SymbolRefAttr>> updates;
    for (const mlir::NamedAttribute &attr : nestedOp->getAttrDictionary())
      if (auto symRef = llvm::dyn_cast<mlir::SymbolRefAttr>(attr.getValue())) {
        if (auto remap = remappings.find(symRef.getLeafReference());
            remap != remappings.end()) {
          mlir::SymbolRefAttr symAttr = mlir::FlatSymbolRefAttr(remap->second);
          if (mlir::isa<mlir::gpu::LaunchFuncOp>(nestedOp))
            symAttr = mlir::SymbolRefAttr::get(
                symRef.getRootReference(),
                {mlir::FlatSymbolRefAttr(remap->second)});
          updates.emplace_back(std::pair<mlir::StringAttr, mlir::SymbolRefAttr>{
              attr.getName(), symAttr});
        }
      }
    for (auto update : updates)
      nestedOp->setAttr(update.first, update.second);
  });
}
