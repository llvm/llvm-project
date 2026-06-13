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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace cir {
namespace direct {

/// Implementation of the dialect interface that converts CIR attributes to LLVM
/// IR metadata.
class CIRDialectLLVMIRTranslationInterface
    : public mlir::LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  mlir::LogicalResult convertOperation(
      mlir::Operation *op, llvm::IRBuilderBase &builder,
      mlir::LLVM::ModuleTranslation &moduleTranslation) const final {

    if (auto cirOp = llvm::dyn_cast<mlir::LLVM::ZeroOp>(op))
      moduleTranslation.mapValue(cirOp.getResult()) =
          llvm::Constant::getNullValue(
              moduleTranslation.convertType(cirOp.getType()));

    return mlir::success();
  }

  /// Any named attribute in the CIR dialect, i.e, with name started with
  /// "cir.", will be handled here.
  virtual mlir::LogicalResult amendOperation(
      mlir::Operation *op, llvm::ArrayRef<llvm::Instruction *> instructions,
      mlir::NamedAttribute attribute,
      mlir::LLVM::ModuleTranslation &moduleTranslation) const override {
    if (auto func = dyn_cast<mlir::LLVM::LLVMFuncOp>(op)) {
      if (mlir::failed(
              amendFunction(func, instructions, attribute, moduleTranslation)))
        return mlir::failure();
    } else if (auto mod = dyn_cast<mlir::ModuleOp>(op)) {
      if (mlir::failed(amendModule(mod, attribute, moduleTranslation)))
        return mlir::failure();
    }
    return mlir::success();
  }

private:
  // Translate CIR function attributes to LLVM function attributes.
  mlir::LogicalResult
  amendFunction(mlir::LLVM::LLVMFuncOp func,
                llvm::ArrayRef<llvm::Instruction *> instructions,
                mlir::NamedAttribute attribute,
                mlir::LLVM::ModuleTranslation &moduleTranslation) const {
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());
    llvm::StringRef attrName = attribute.getName().strref();

    // Strip the "cir." prefix to get the LLVM attribute name.
    llvm::StringRef llvmAttrName = attrName.substr(strlen("cir."));
    if (auto clArgMetadata = mlir::dyn_cast<cir::OpenCLKernelArgMetadataAttr>(
            attribute.getValue())) {
      emitOpenCLKernelArgMetadata(clArgMetadata, llvmFunc, moduleTranslation);
    } else if (auto strAttr =
                   mlir::dyn_cast<mlir::StringAttr>(attribute.getValue())) {
      llvmFunc->addFnAttr(llvmAttrName, strAttr.getValue());
    }
    return mlir::success();
  }

  void emitOpenCLKernelArgMetadata(
      cir::OpenCLKernelArgMetadataAttr clArgMetadata, llvm::Function *llvmFunc,
      mlir::LLVM::ModuleTranslation &moduleTranslation) const {
    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();

    SmallVector<llvm::Metadata *, 8> addressQuals;

    auto getOpenCLArgInfoAddressSpace = [](cir::LangAddressSpace addressSpace) {
      switch (addressSpace) {
      case cir::LangAddressSpace::Default:
      case cir::LangAddressSpace::OffloadPrivate:
        return 0u;
      case cir::LangAddressSpace::OffloadGlobal:
        return 1u;
      case cir::LangAddressSpace::OffloadConstant:
        return 2u;
      case cir::LangAddressSpace::OffloadLocal:
        return 3u;
      case cir::LangAddressSpace::OffloadGeneric:
        return 4u;
      case cir::LangAddressSpace::OffloadGlobalDevice:
        return 5u;
      case cir::LangAddressSpace::OffloadGlobalHost:
        return 6u;
      }
      llvm_unreachable("unknown CIR language address space");
    };

    for (cir::LangAddressSpaceAttr addressSpace :
         clArgMetadata.getAddrSpace().getAsRange<cir::LangAddressSpaceAttr>()) {
      addressQuals.push_back(
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
              llvm::IntegerType::get(llvmContext, 32),
              getOpenCLArgInfoAddressSpace(addressSpace.getValue()))));
    }

    llvmFunc->setMetadata("kernel_arg_addr_space",
                          llvm::MDNode::get(llvmContext, addressQuals));
    llvmFunc->setMetadata(
        "kernel_arg_access_qual",
        getStringArrayMetadataNode(llvmContext, clArgMetadata.getAccessQual()));
    llvmFunc->setMetadata(
        "kernel_arg_type",
        getStringArrayMetadataNode(llvmContext, clArgMetadata.getType()));
    llvmFunc->setMetadata(
        "kernel_arg_base_type",
        getStringArrayMetadataNode(llvmContext, clArgMetadata.getBaseType()));
    llvmFunc->setMetadata(
        "kernel_arg_type_qual",
        getStringArrayMetadataNode(llvmContext, clArgMetadata.getTypeQual()));
    if (clArgMetadata.getName())
      llvmFunc->setMetadata(
          "kernel_arg_name",
          getStringArrayMetadataNode(llvmContext, clArgMetadata.getName()));
  }

  llvm::MDNode *getStringArrayMetadataNode(llvm::LLVMContext &llvmContext,
                                           mlir::ArrayAttr attrs) const {
    SmallVector<llvm::Metadata *, 8> metadata;
    for (mlir::StringAttr attr : attrs.getAsRange<mlir::StringAttr>())
      metadata.push_back(llvm::MDString::get(llvmContext, attr.getValue()));
    return llvm::MDNode::get(llvmContext, metadata);
  }

  // Translate CIR's module attributes to LLVM's module metadata
  mlir::LogicalResult
  amendModule(mlir::ModuleOp mod, mlir::NamedAttribute attribute,
              mlir::LLVM::ModuleTranslation &moduleTranslation) const {
    llvm::Module *llvmModule = moduleTranslation.getLLVMModule();
    llvm::LLVMContext &llvmContext = llvmModule->getContext();

    if (attribute.getName() == "cir.amdhsa_code_object_version") {
      if (auto intAttr =
              mlir::dyn_cast<mlir::IntegerAttr>(attribute.getValue())) {
        llvmModule->addModuleFlag(llvm::Module::Error,
                                  "amdhsa_code_object_version",
                                  static_cast<uint32_t>(intAttr.getInt()));
      }
    }

    if (attribute.getName() == "cir.amdgpu_printf_kind") {
      if (auto strAttr =
              mlir::dyn_cast<mlir::StringAttr>(attribute.getValue())) {
        llvm::MDString *mdStr =
            llvm::MDString::get(llvmContext, strAttr.getValue());
        llvmModule->addModuleFlag(llvm::Module::Error, "amdgpu_printf_kind",
                                  mdStr);
      }
    }

    return mlir::success();
  }
};

void registerCIRDialectTranslation(mlir::DialectRegistry &registry) {
  registry.insert<cir::CIRDialect>();
  registry.addExtension(+[](mlir::MLIRContext *ctx, cir::CIRDialect *dialect) {
    dialect->addInterfaces<CIRDialectLLVMIRTranslationInterface>();
  });
}

} // namespace direct
} // namespace cir

namespace mlir {
void registerCIRDialectTranslation(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  cir::direct::registerCIRDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
} // namespace mlir
