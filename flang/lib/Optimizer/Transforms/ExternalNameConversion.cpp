//===- ExternalNameConversion.cpp -- convert name with external convention ===//
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
#include "flang/Support/Fortran.h"
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

/// Process a symbol reference and return the updated symbol reference if
/// needed.
std::optional<mlir::SymbolRefAttr>
processSymbolRef(mlir::SymbolRefAttr symRef, mlir::Operation *nestedOp,
                 const llvm::DenseMap<mlir::StringAttr, mlir::FlatSymbolRefAttr>
                     &remappings) {
  if (auto remap = remappings.find(symRef.getLeafReference());
      remap != remappings.end()) {
    mlir::SymbolRefAttr symAttr = mlir::FlatSymbolRefAttr(remap->second);
    if (mlir::isa<mlir::gpu::LaunchFuncOp>(nestedOp))
      symAttr = mlir::SymbolRefAttr::get(
          symRef.getRootReference(), {mlir::FlatSymbolRefAttr(remap->second)});
    return symAttr;
  }
  return std::nullopt;
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
    llvm::SmallVector<std::pair<mlir::StringAttr, mlir::SymbolRefAttr>>
        symRefUpdates;
    llvm::SmallVector<std::pair<mlir::StringAttr, mlir::ArrayAttr>>
        arrayUpdates;
    for (const mlir::NamedAttribute &attr : nestedOp->getAttrDictionary())
      if (auto symRef = llvm::dyn_cast<mlir::SymbolRefAttr>(attr.getValue())) {
        if (auto newSymRef = processSymbolRef(symRef, nestedOp, remappings))
          symRefUpdates.emplace_back(
              std::pair<mlir::StringAttr, mlir::SymbolRefAttr>{attr.getName(),
                                                               *newSymRef});
      } else if (auto arrayAttr =
                     llvm::dyn_cast<mlir::ArrayAttr>(attr.getValue())) {
        llvm::SmallVector<mlir::Attribute> symbolRefs;
        for (auto element : arrayAttr) {
          if (!element) {
            symbolRefs.push_back(element);
            continue;
          }
          auto symRef = llvm::dyn_cast<mlir::SymbolRefAttr>(element);
          std::optional<mlir::SymbolRefAttr> updatedSymRef;
          if (symRef)
            updatedSymRef = processSymbolRef(symRef, nestedOp, remappings);
          if (!symRef || !updatedSymRef)
            symbolRefs.push_back(element);
          else
            symbolRefs.push_back(*updatedSymRef);
        }
        arrayUpdates.push_back(std::make_pair(
            attr.getName(),
            mlir::ArrayAttr::get(nestedOp->getContext(), symbolRefs)));
      }
    for (auto update : symRefUpdates)
      nestedOp->setAttr(update.first, update.second);
    for (auto update : arrayUpdates)
      nestedOp->setAttr(update.first, update.second);
  });
}
