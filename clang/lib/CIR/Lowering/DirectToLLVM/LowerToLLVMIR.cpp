//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR attributes and operations directly to
// LLVMIR.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/DialectRegistry.h"
#include "aiir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "aiir/Target/LLVMIR/ModuleTranslation.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;

namespace cir {
namespace direct {

/// Implementation of the dialect interface that converts CIR attributes to LLVM
/// IR metadata.
class CIRDialectLLVMIRTranslationInterface
    : public aiir::LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  aiir::LogicalResult convertOperation(
      aiir::Operation *op, llvm::IRBuilderBase &builder,
      aiir::LLVM::ModuleTranslation &moduleTranslation) const final {

    if (auto cirOp = llvm::dyn_cast<aiir::LLVM::ZeroOp>(op))
      moduleTranslation.mapValue(cirOp.getResult()) =
          llvm::Constant::getNullValue(
              moduleTranslation.convertType(cirOp.getType()));

    return aiir::success();
  }

  /// Any named attribute in the CIR dialect, i.e, with name started with
  /// "cir.", will be handled here.
  virtual aiir::LogicalResult amendOperation(
      aiir::Operation *op, llvm::ArrayRef<llvm::Instruction *> instructions,
      aiir::NamedAttribute attribute,
      aiir::LLVM::ModuleTranslation &moduleTranslation) const override {
    if (auto func = dyn_cast<aiir::LLVM::LLVMFuncOp>(op)) {
      if (aiir::failed(
              amendFunction(func, instructions, attribute, moduleTranslation)))
        return aiir::failure();
    } else if (auto mod = dyn_cast<aiir::ModuleOp>(op)) {
      if (aiir::failed(amendModule(mod, attribute, moduleTranslation)))
        return aiir::failure();
    }
    return aiir::success();
  }

private:
  // Translate CIR function attributes to LLVM function attributes.
  aiir::LogicalResult
  amendFunction(aiir::LLVM::LLVMFuncOp func,
                llvm::ArrayRef<llvm::Instruction *> instructions,
                aiir::NamedAttribute attribute,
                aiir::LLVM::ModuleTranslation &moduleTranslation) const {
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());
    llvm::StringRef attrName = attribute.getName().strref();

    // Strip the "cir." prefix to get the LLVM attribute name.
    llvm::StringRef llvmAttrName = attrName.substr(strlen("cir."));
    if (auto strAttr = aiir::dyn_cast<aiir::StringAttr>(attribute.getValue()))
      llvmFunc->addFnAttr(llvmAttrName, strAttr.getValue());
    return aiir::success();
  }

  // Translate CIR's module attributes to LLVM's module metadata
  aiir::LogicalResult
  amendModule(aiir::ModuleOp mod, aiir::NamedAttribute attribute,
              aiir::LLVM::ModuleTranslation &moduleTranslation) const {
    llvm::Module *llvmModule = moduleTranslation.getLLVMModule();
    llvm::LLVMContext &llvmContext = llvmModule->getContext();

    if (attribute.getName() == "cir.amdhsa_code_object_version") {
      if (auto intAttr =
              aiir::dyn_cast<aiir::IntegerAttr>(attribute.getValue())) {
        llvmModule->addModuleFlag(llvm::Module::Error,
                                  "amdhsa_code_object_version",
                                  static_cast<uint32_t>(intAttr.getInt()));
      }
    }

    if (attribute.getName() == "cir.amdgpu_printf_kind") {
      if (auto strAttr =
              aiir::dyn_cast<aiir::StringAttr>(attribute.getValue())) {
        llvm::MDString *mdStr =
            llvm::MDString::get(llvmContext, strAttr.getValue());
        llvmModule->addModuleFlag(llvm::Module::Error, "amdgpu_printf_kind",
                                  mdStr);
      }
    }

    return aiir::success();
  }
};

void registerCIRDialectTranslation(aiir::DialectRegistry &registry) {
  registry.insert<cir::CIRDialect>();
  registry.addExtension(+[](aiir::AIIRContext *ctx, cir::CIRDialect *dialect) {
    dialect->addInterfaces<CIRDialectLLVMIRTranslationInterface>();
  });
}

} // namespace direct
} // namespace cir

namespace aiir {
void registerCIRDialectTranslation(aiir::AIIRContext &context) {
  aiir::DialectRegistry registry;
  cir::direct::registerCIRDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
} // namespace aiir
