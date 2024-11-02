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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace fir {
#define GEN_PASS_DEF_COMPILERGENERATEDNAMESCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

namespace {

class CompilerGeneratedNamesConversionPass
    : public fir::impl::CompilerGeneratedNamesConversionBase<
          CompilerGeneratedNamesConversionPass> {
public:
  using CompilerGeneratedNamesConversionBase<
      CompilerGeneratedNamesConversionPass>::
      CompilerGeneratedNamesConversionBase;

  mlir::ModuleOp getModule() { return getOperation(); }
  void runOnOperation() override;
};
} // namespace

void CompilerGeneratedNamesConversionPass::runOnOperation() {
  auto op = getOperation();
  auto *context = &getContext();

  llvm::DenseMap<mlir::StringAttr, mlir::FlatSymbolRefAttr> remappings;
  for (auto &funcOrGlobal : op->getRegion(0).front()) {
    if (llvm::isa<mlir::func::FuncOp>(funcOrGlobal) ||
        llvm::isa<fir::GlobalOp>(funcOrGlobal)) {
      auto symName = funcOrGlobal.getAttrOfType<mlir::StringAttr>(
          mlir::SymbolTable::getSymbolAttrName());
      auto deconstructedName = fir::NameUniquer::deconstruct(symName);
      if (deconstructedName.first != fir::NameUniquer::NameKind::NOT_UNIQUED &&
          !fir::NameUniquer::isExternalFacingUniquedName(deconstructedName)) {
        std::string newName =
            fir::NameUniquer::replaceSpecialSymbols(symName.getValue().str());
        if (newName != symName) {
          auto newAttr = mlir::StringAttr::get(context, newName);
          mlir::SymbolTable::setSymbolName(&funcOrGlobal, newAttr);
          auto newSymRef = mlir::FlatSymbolRefAttr::get(newAttr);
          remappings.try_emplace(symName, newSymRef);
        }
      }
    }
  }

  if (remappings.empty())
    return;

  // Update all uses of the functions and globals that have been renamed.
  op.walk([&remappings](mlir::Operation *nestedOp) {
    llvm::SmallVector<std::pair<mlir::StringAttr, mlir::SymbolRefAttr>> updates;
    for (const mlir::NamedAttribute &attr : nestedOp->getAttrDictionary())
      if (auto symRef = llvm::dyn_cast<mlir::SymbolRefAttr>(attr.getValue()))
        if (auto remap = remappings.find(symRef.getRootReference());
            remap != remappings.end())
          updates.emplace_back(std::pair<mlir::StringAttr, mlir::SymbolRefAttr>{
              attr.getName(), mlir::SymbolRefAttr(remap->second)});
    for (auto update : updates)
      nestedOp->setAttr(update.first, update.second);
  });
}
