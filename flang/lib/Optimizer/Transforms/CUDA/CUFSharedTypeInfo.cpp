//===-- CUFSharedTypeInfo.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/xxhash.h"

namespace fir {
#define GEN_PASS_DEF_CUFSHAREDTYPEINFO
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {

/// Return a tag identifying the translation unit of \p mod.
///
/// Without relocatable device code, every unit has its own device module, and
/// the CUDA runtime only populates the first managed variable registered under
/// a given name. The managed pointers created here must therefore have a name
/// unique to their unit.
static std::string getUnitTag(mlir::ModuleOp mod) {
  std::string key;
  llvm::raw_string_ostream os(key);
  if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(mod.getLoc()))
    os << fileLoc.getFilename().getValue();
  for (mlir::Operation &op : mod.getBody()->getOperations())
    if (auto sym = mlir::dyn_cast<mlir::SymbolOpInterface>(op))
      os << ';' << sym.getName();
  return llvm::utohexstr(llvm::xxh3_64bits(key), /*LowerCase=*/true);
}

class CUFSharedTypeInfo
    : public fir::impl::CUFSharedTypeInfoBase<CUFSharedTypeInfo> {
public:
  using CUFSharedTypeInfoBase::CUFSharedTypeInfoBase;

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::MLIRContext *ctx = mod.getContext();
    mlir::SymbolTable symTab(mod);
    mlir::StringAttr declareAttrName =
        mlir::StringAttr::get(ctx, mlir::acc::getDeclareAttrName());
    mlir::StringAttr section =
        mlir::StringAttr::get(ctx, cudaSharedTypeInfoSection);

    llvm::SmallVector<fir::GlobalOp> typeDescs;
    for (fir::GlobalOp global : mod.getOps<fir::GlobalOp>()) {
      if (!cuf::isTypeInfoGlobal(global))
        continue;
      // The type information is no longer copied to the device.
      global->removeAttr(declareAttrName);
      // Constant data would be placed in a read-only section, which not every
      // GPU can register. Every type-info global must land in the same
      // section, and a section cannot mix read-only and writable data.
      if (global.isInitialized()) {
        global.removeConstantAttr();
        global.setSectionAttr(section);
      }
      if (cuf::isTypeDescriptorGlobal(global))
        typeDescs.push_back(global);
    }

    auto gpuMod = symTab.lookup<mlir::gpu::GPUModuleOp>(cudaDeviceModuleName);
    if (!gpuMod)
      return;
    mlir::SymbolTable gpuSymTab(gpuMod);
    for (fir::GlobalOp global : gpuMod.getOps<fir::GlobalOp>())
      if (cuf::isTypeInfoGlobal(global))
        global->removeAttr(declareAttrName);

    // Device code reads each type descriptor through a managed pointer that
    // the CUDA Fortran constructor sets to the host address. A single managed
    // allocation is valid on every device.
    //
    // The pointer names and the dictionary keys use the assembly form of the
    // type descriptor name. CompilerGeneratedNamesConversion does not update
    // names nested in the dictionary, and leaves names in this form unchanged.
    std::string unitTag = getUnitTag(mod);
    mlir::OpBuilder builder(ctx);
    auto ptrTy = fir::LLVMPointerType::get(ctx, mlir::IntegerType::get(ctx, 8));
    auto managed =
        cuf::DataAttributeAttr::get(ctx, cuf::DataAttribute::Managed);
    llvm::SmallVector<mlir::NamedAttribute> sharedTypeDescs;
    for (fir::GlobalOp typeDesc : typeDescs) {
      if (!gpuSymTab.lookup(typeDesc.getSymName()))
        continue;
      std::string typeDescName =
          fir::NameUniquer::replaceSpecialSymbols(typeDesc.getSymName().str());
      std::string ptrName = typeDescName + "Xhostaddr" + unitTag;
      builder.setInsertionPointAfter(typeDesc);
      auto ptrGlobal = fir::GlobalOp::create(
          builder, typeDesc.getLoc(), ptrName, /*isConstant=*/false,
          /*isTarget=*/false, ptrTy, mlir::Attribute{},
          /*linkage=*/fir::LinkageAttr{});
      ptrGlobal.setDataAttrAttr(managed);
      mlir::Block *block = builder.createBlock(&ptrGlobal.getRegion());
      builder.setInsertionPointToStart(block);
      mlir::Value zero = fir::ZeroOp::create(builder, typeDesc.getLoc(), ptrTy);
      fir::HasValueOp::create(builder, typeDesc.getLoc(), zero);

      gpuSymTab.insert(ptrGlobal->clone());
      ptrGlobal->setAttr(
          cudaHostTypeDescAttrName,
          mlir::FlatSymbolRefAttr::get(typeDesc.getSymNameAttr()));
      sharedTypeDescs.emplace_back(
          mlir::StringAttr::get(ctx, typeDescName),
          mlir::FlatSymbolRefAttr::get(ptrGlobal.getSymNameAttr()));
    }
    if (!sharedTypeDescs.empty())
      gpuMod->setAttr(cudaSharedTypeDescsAttrName,
                      mlir::DictionaryAttr::get(ctx, sharedTypeDescs));
  }
};

} // namespace
