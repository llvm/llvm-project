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
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/IR/Attributes.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Pass/Pass.h"

namespace fir {
#define GEN_PASS_DEF_EXTERNALNAMECONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace aiir;

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
std::optional<aiir::SymbolRefAttr>
processSymbolRef(aiir::SymbolRefAttr symRef, aiir::Operation *nestedOp,
                 const llvm::DenseMap<aiir::StringAttr, aiir::FlatSymbolRefAttr>
                     &remappings) {
  if (auto remap = remappings.find(symRef.getLeafReference());
      remap != remappings.end()) {
    aiir::SymbolRefAttr symAttr = aiir::FlatSymbolRefAttr(remap->second);
    if (aiir::isa<aiir::gpu::LaunchFuncOp>(nestedOp))
      symAttr = aiir::SymbolRefAttr::get(
          symRef.getRootReference(), {aiir::FlatSymbolRefAttr(remap->second)});
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

  aiir::ModuleOp getModule() { return getOperation(); }
  void runOnOperation() override;
};
} // namespace

void ExternalNameConversionPass::runOnOperation() {
  auto op = getOperation();
  auto *context = &getContext();

  llvm::DenseMap<aiir::StringAttr, aiir::FlatSymbolRefAttr> remappings;
  aiir::SymbolTable symbolTable(op);

  auto processFctOrGlobal = [&](aiir::Operation &funcOrGlobal) {
    auto symName = funcOrGlobal.getAttrOfType<aiir::StringAttr>(
        aiir::SymbolTable::getSymbolAttrName());
    auto deconstructedName = fir::NameUniquer::deconstruct(symName);
    if (fir::NameUniquer::isExternalFacingUniquedName(deconstructedName)) {
      // Check if this is a private function that would conflict with a common
      // block and get its mangled name.
      if (auto funcOp = llvm::dyn_cast<aiir::func::FuncOp>(funcOrGlobal)) {
        if (funcOp.isPrivate()) {
          std::string mangledName =
              mangleExternalName(deconstructedName, appendUnderscoreOpt);
          auto mod = funcOp->getParentOfType<aiir::ModuleOp>();
          bool hasConflictingCommonBlock = false;

          // Check if any existing global has the same mangled name.
          if (symbolTable.lookup<fir::GlobalOp>(mangledName))
            hasConflictingCommonBlock = true;

          // Skip externalization if the function has a conflicting common block
          // and is not directly called (i.e. procedure pointers or type
          // specifications)
          if (hasConflictingCommonBlock) {
            bool isDirectlyCalled = false;
            std::optional<SymbolTable::UseRange> uses =
                funcOp.getSymbolUses(mod);
            if (uses.has_value()) {
              for (auto use : *uses) {
                aiir::Operation *user = use.getUser();
                if (aiir::isa<fir::CallOp>(user) ||
                    aiir::isa<aiir::func::CallOp>(user)) {
                  isDirectlyCalled = true;
                  break;
                }
              }
            }
            if (!isDirectlyCalled)
              return;
          }
        }
      }

      auto newName = mangleExternalName(deconstructedName, appendUnderscoreOpt);
      auto newAttr = aiir::StringAttr::get(context, newName);
      aiir::SymbolTable::setSymbolName(&funcOrGlobal, newAttr);
      auto newSymRef = aiir::FlatSymbolRefAttr::get(newAttr);
      remappings.try_emplace(symName, newSymRef);
      if (llvm::isa<aiir::func::FuncOp>(funcOrGlobal))
        funcOrGlobal.setAttr(fir::getInternalFuncNameAttrName(), symName);
    }
  };

  auto renameFuncOrGlobalInModule = [&](aiir::Operation *module) {
    for (auto &op : module->getRegion(0).front()) {
      if (aiir::isa<aiir::func::FuncOp, fir::GlobalOp>(op)) {
        processFctOrGlobal(op);
      } else if (auto gpuMod = aiir::dyn_cast<aiir::gpu::GPUModuleOp>(op)) {
        for (auto &gpuOp : gpuMod.getBodyRegion().front())
          if (aiir::isa<aiir::func::FuncOp, fir::GlobalOp,
                        aiir::gpu::GPUFuncOp>(gpuOp))
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
  op.walk([&remappings](aiir::Operation *nestedOp) {
    llvm::SmallVector<std::pair<aiir::StringAttr, aiir::SymbolRefAttr>>
        symRefUpdates;
    llvm::SmallVector<std::pair<aiir::StringAttr, aiir::ArrayAttr>>
        arrayUpdates;
    for (const aiir::NamedAttribute &attr : nestedOp->getAttrDictionary())
      if (auto symRef = llvm::dyn_cast<aiir::SymbolRefAttr>(attr.getValue())) {
        if (auto newSymRef = processSymbolRef(symRef, nestedOp, remappings))
          symRefUpdates.emplace_back(
              std::pair<aiir::StringAttr, aiir::SymbolRefAttr>{attr.getName(),
                                                               *newSymRef});
      } else if (auto arrayAttr =
                     llvm::dyn_cast<aiir::ArrayAttr>(attr.getValue())) {
        llvm::SmallVector<aiir::Attribute> symbolRefs;
        for (auto element : arrayAttr) {
          if (!element) {
            symbolRefs.push_back(element);
            continue;
          }
          auto symRef = llvm::dyn_cast<aiir::SymbolRefAttr>(element);
          std::optional<aiir::SymbolRefAttr> updatedSymRef;
          if (symRef)
            updatedSymRef = processSymbolRef(symRef, nestedOp, remappings);
          if (!symRef || !updatedSymRef)
            symbolRefs.push_back(element);
          else
            symbolRefs.push_back(*updatedSymRef);
        }
        arrayUpdates.push_back(std::make_pair(
            attr.getName(),
            aiir::ArrayAttr::get(nestedOp->getContext(), symbolRefs)));
      }
    for (auto update : symRefUpdates)
      nestedOp->setAttr(update.first, update.second);
    for (auto update : arrayUpdates)
      nestedOp->setAttr(update.first, update.second);
  });
}
