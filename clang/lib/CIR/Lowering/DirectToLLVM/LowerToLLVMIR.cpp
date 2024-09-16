//====- LoweToLLVMIR.cpp - Lowering CIR attributes to LLVMIR ---------===//
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
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;

namespace cir {
namespace direct {

/// Implementation of the dialect interface that converts CIR attributes to LLVM
/// IR metadata.
class CIRDialectLLVMIRTranslationInterface
    : public mlir::LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Any named attribute in the CIR dialect, i.e, with name started with
  /// "cir.", will be handled here.
  virtual mlir::LogicalResult amendOperation(
      mlir::Operation *op, llvm::ArrayRef<llvm::Instruction *> instructions,
      mlir::NamedAttribute attribute,
      mlir::LLVM::ModuleTranslation &moduleTranslation) const override {
    if (auto func = dyn_cast<mlir::LLVM::LLVMFuncOp>(op)) {
      amendFunction(func, instructions, attribute, moduleTranslation);
    } else if (auto mod = dyn_cast<mlir::ModuleOp>(op)) {
      amendModule(mod, attribute, moduleTranslation);
    }
    return mlir::success();
  }

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

private:
  // Translate CIR's module attributes to LLVM's module metadata
  void amendModule(mlir::ModuleOp module, mlir::NamedAttribute attribute,
                   mlir::LLVM::ModuleTranslation &moduleTranslation) const {
    llvm::Module *llvmModule = moduleTranslation.getLLVMModule();
    llvm::LLVMContext &llvmContext = llvmModule->getContext();

    if (auto openclVersionAttr = mlir::dyn_cast<mlir::cir::OpenCLVersionAttr>(
            attribute.getValue())) {
      auto *int32Ty = llvm::IntegerType::get(llvmContext, 32);
      llvm::Metadata *oclVerElts[] = {
          llvm::ConstantAsMetadata::get(
              llvm::ConstantInt::get(int32Ty, openclVersionAttr.getMajor())),
          llvm::ConstantAsMetadata::get(
              llvm::ConstantInt::get(int32Ty, openclVersionAttr.getMinor()))};
      llvm::NamedMDNode *oclVerMD =
          llvmModule->getOrInsertNamedMetadata("opencl.ocl.version");
      oclVerMD->addOperand(llvm::MDNode::get(llvmContext, oclVerElts));
    }

    // Drop ammended CIR attribute from LLVM op.
    module->removeAttr(attribute.getName());
  }

  // Translate CIR's extra function attributes to LLVM's function attributes.
  void amendFunction(mlir::LLVM::LLVMFuncOp func,
                     llvm::ArrayRef<llvm::Instruction *> instructions,
                     mlir::NamedAttribute attribute,
                     mlir::LLVM::ModuleTranslation &moduleTranslation) const {
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());
    if (auto extraAttr = mlir::dyn_cast<mlir::cir::ExtraFuncAttributesAttr>(
            attribute.getValue())) {
      for (auto attr : extraAttr.getElements()) {
        if (auto inlineAttr =
                mlir::dyn_cast<mlir::cir::InlineAttr>(attr.getValue())) {
          if (inlineAttr.isNoInline())
            llvmFunc->addFnAttr(llvm::Attribute::NoInline);
          else if (inlineAttr.isAlwaysInline())
            llvmFunc->addFnAttr(llvm::Attribute::AlwaysInline);
          else if (inlineAttr.isInlineHint())
            llvmFunc->addFnAttr(llvm::Attribute::InlineHint);
          else
            llvm_unreachable("Unknown inline kind");
        } else if (mlir::dyn_cast<mlir::cir::OptNoneAttr>(attr.getValue())) {
          llvmFunc->addFnAttr(llvm::Attribute::OptimizeNone);
        } else if (mlir::dyn_cast<mlir::cir::NoThrowAttr>(attr.getValue())) {
          llvmFunc->addFnAttr(llvm::Attribute::NoUnwind);
        } else if (mlir::dyn_cast<mlir::cir::ConvergentAttr>(attr.getValue())) {
          llvmFunc->addFnAttr(llvm::Attribute::Convergent);
        } else if (auto clKernelMetadata =
                       mlir::dyn_cast<mlir::cir::OpenCLKernelMetadataAttr>(
                           attr.getValue())) {
          emitOpenCLKernelMetadata(clKernelMetadata, llvmFunc,
                                   moduleTranslation);
        } else if (auto clArgMetadata =
                       mlir::dyn_cast<mlir::cir::OpenCLKernelArgMetadataAttr>(
                           attr.getValue())) {
          emitOpenCLKernelArgMetadata(clArgMetadata, func.getNumArguments(),
                                      llvmFunc, moduleTranslation);
        }
      }
    }

    // Drop ammended CIR attribute from LLVM op.
    func->removeAttr(attribute.getName());
  }

  void emitOpenCLKernelMetadata(
      mlir::cir::OpenCLKernelMetadataAttr clKernelMetadata,
      llvm::Function *llvmFunc,
      mlir::LLVM::ModuleTranslation &moduleTranslation) const {
    auto &vmCtx = moduleTranslation.getLLVMContext();

    auto lowerArrayAttr = [&](mlir::ArrayAttr arrayAttr) {
      llvm::SmallVector<llvm::Metadata *, 3> attrMDArgs;
      for (mlir::Attribute attr : arrayAttr) {
        int64_t value = mlir::cast<mlir::IntegerAttr>(attr).getInt();
        attrMDArgs.push_back(
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::IntegerType::get(vmCtx, 32), llvm::APInt(32, value))));
      }
      return llvm::MDNode::get(vmCtx, attrMDArgs);
    };

    if (auto workGroupSizeHint = clKernelMetadata.getWorkGroupSizeHint()) {
      llvmFunc->setMetadata("work_group_size_hint",
                            lowerArrayAttr(workGroupSizeHint));
    }

    if (auto reqdWorkGroupSize = clKernelMetadata.getReqdWorkGroupSize()) {
      llvmFunc->setMetadata("reqd_work_group_size",
                            lowerArrayAttr(reqdWorkGroupSize));
    }

    if (auto vecTypeHint = clKernelMetadata.getVecTypeHint()) {
      auto hintQTy = vecTypeHint.getValue();
      bool isSignedInteger = *clKernelMetadata.getVecTypeHintSignedness();
      llvm::Metadata *attrMDArgs[] = {
          llvm::ConstantAsMetadata::get(
              llvm::UndefValue::get(moduleTranslation.convertType(hintQTy))),
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
              llvm::IntegerType::get(vmCtx, 32),
              llvm::APInt(32, (uint64_t)(isSignedInteger ? 1 : 0))))};
      llvmFunc->setMetadata("vec_type_hint",
                            llvm::MDNode::get(vmCtx, attrMDArgs));
    }

    if (auto intelReqdSubgroupSize =
            clKernelMetadata.getIntelReqdSubGroupSize()) {
      int64_t reqdSubgroupSize = intelReqdSubgroupSize.getInt();
      llvm::Metadata *attrMDArgs[] = {
          llvm::ConstantAsMetadata::get(
              llvm::ConstantInt::get(llvm::IntegerType::get(vmCtx, 32),
                                     llvm::APInt(32, reqdSubgroupSize))),
      };
      llvmFunc->setMetadata("intel_reqd_sub_group_size",
                            llvm::MDNode::get(vmCtx, attrMDArgs));
    }
  }

  void emitOpenCLKernelArgMetadata(
      mlir::cir::OpenCLKernelArgMetadataAttr clArgMetadata, unsigned numArgs,
      llvm::Function *llvmFunc,
      mlir::LLVM::ModuleTranslation &moduleTranslation) const {
    auto &vmCtx = moduleTranslation.getLLVMContext();

    // MDNode for the kernel argument address space qualifiers.
    SmallVector<llvm::Metadata *, 8> addressQuals;

    // MDNode for the kernel argument access qualifiers (images only).
    SmallVector<llvm::Metadata *, 8> accessQuals;

    // MDNode for the kernel argument type names.
    SmallVector<llvm::Metadata *, 8> argTypeNames;

    // MDNode for the kernel argument base type names.
    SmallVector<llvm::Metadata *, 8> argBaseTypeNames;

    // MDNode for the kernel argument type qualifiers.
    SmallVector<llvm::Metadata *, 8> argTypeQuals;

    // MDNode for the kernel argument names.
    SmallVector<llvm::Metadata *, 8> argNames;

    auto lowerStringAttr = [&](mlir::Attribute strAttr) {
      return llvm::MDString::get(
          vmCtx, mlir::cast<mlir::StringAttr>(strAttr).getValue());
    };

    bool shouldEmitArgName = !!clArgMetadata.getName();
    
    auto addressSpaceValues =
        clArgMetadata.getAddrSpace().getAsValueRange<mlir::IntegerAttr>();

    for (auto &&[i, addrSpace] : llvm::enumerate(addressSpaceValues)) {
      // Address space qualifier.
      addressQuals.push_back(
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
              llvm::IntegerType::get(vmCtx, 32), addrSpace)));

      // Access qualifier.
      accessQuals.push_back(lowerStringAttr(clArgMetadata.getAccessQual()[i]));

      // Type name.
      argTypeNames.push_back(lowerStringAttr(clArgMetadata.getType()[i]));

      // Base type name.
      argBaseTypeNames.push_back(
          lowerStringAttr(clArgMetadata.getBaseType()[i]));

      // Type qualifier.
      argTypeQuals.push_back(lowerStringAttr(clArgMetadata.getTypeQual()[i]));

      // Argument name.
      if (shouldEmitArgName)
        argNames.push_back(lowerStringAttr(clArgMetadata.getName()[i]));
    }

    llvmFunc->setMetadata("kernel_arg_addr_space",
                          llvm::MDNode::get(vmCtx, addressQuals));
    llvmFunc->setMetadata("kernel_arg_access_qual",
                          llvm::MDNode::get(vmCtx, accessQuals));
    llvmFunc->setMetadata("kernel_arg_type",
                          llvm::MDNode::get(vmCtx, argTypeNames));
    llvmFunc->setMetadata("kernel_arg_base_type",
                          llvm::MDNode::get(vmCtx, argBaseTypeNames));
    llvmFunc->setMetadata("kernel_arg_type_qual",
                          llvm::MDNode::get(vmCtx, argTypeQuals));
    if (shouldEmitArgName)
      llvmFunc->setMetadata("kernel_arg_name",
                            llvm::MDNode::get(vmCtx, argNames));
  }
};

void registerCIRDialectTranslation(mlir::DialectRegistry &registry) {
  registry.insert<mlir::cir::CIRDialect>();
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::cir::CIRDialect *dialect) {
        dialect->addInterfaces<CIRDialectLLVMIRTranslationInterface>();
      });
}

void registerCIRDialectTranslation(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerCIRDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
} // namespace direct
} // namespace cir
