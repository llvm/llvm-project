//===- CUFToLLVMIRTranslation.cpp - Translate CUF dialect to LLVM IR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR CUF dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/CUF/CUFToLLVMIRTranslation.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Runtime/entry-names.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

namespace {

LogicalResult registerModule(cuf::RegisterModuleOp op,
                             llvm::IRBuilderBase &builder,
                             LLVM::ModuleTranslation &moduleTranslation) {
  std::string binaryIdentifier =
      op.getName().getLeafReference().str() + "_bin_cst";
  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::Value *binary = module->getGlobalVariable(binaryIdentifier, true);
  if (!binary)
    return op.emitError() << "Couldn't find the binary: " << binaryIdentifier;

  llvm::Type *ptrTy = builder.getPtrTy(0);
  llvm::FunctionCallee fct = module->getOrInsertFunction(
      RTNAME_STRING(CUFRegisterModule),
      llvm::FunctionType::get(ptrTy, ArrayRef<llvm::Type *>({ptrTy}), false));
  auto *handle = builder.CreateCall(fct, {binary});
  moduleTranslation.mapValue(op->getResults().front()) = handle;
  return mlir::success();
}

llvm::Value *getOrCreateFunctionName(llvm::Module *module,
                                     llvm::IRBuilderBase &builder,
                                     llvm::StringRef moduleName,
                                     llvm::StringRef kernelName) {
  std::string globalName =
      std::string(llvm::formatv("{0}_{1}_kernel_name", moduleName, kernelName));

  if (llvm::GlobalVariable *gv = module->getGlobalVariable(globalName))
    return gv;

  return builder.CreateGlobalString(kernelName, globalName);
}

LogicalResult registerKernel(cuf::RegisterKernelOp op,
                             llvm::IRBuilderBase &builder,
                             LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *module = moduleTranslation.getLLVMModule();
  llvm::Type *ptrTy = builder.getPtrTy(0);
  llvm::FunctionCallee fct = module->getOrInsertFunction(
      RTNAME_STRING(CUFRegisterFunction),
      llvm::FunctionType::get(
          ptrTy, ArrayRef<llvm::Type *>({ptrTy, ptrTy, ptrTy}), false));
  llvm::Value *modulePtr = moduleTranslation.lookupValue(op.getModulePtr());
  if (!modulePtr)
    return op.emitError() << "Couldn't find the module ptr";
  llvm::Function *fctSym =
      moduleTranslation.lookupFunction(op.getKernelName().str());
  if (!fctSym)
    return op.emitError() << "Couldn't find kernel name symbol: "
                          << op.getKernelName().str();
  builder.CreateCall(fct, {modulePtr, fctSym,
                           getOrCreateFunctionName(
                               module, builder, op.getKernelModuleName().str(),
                               op.getKernelName().str())});
  return mlir::success();
}

class CUFDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *operation, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const override {
    return llvm::TypeSwitch<Operation *, LogicalResult>(operation)
        .Case([&](cuf::RegisterModuleOp op) {
          return registerModule(op, builder, moduleTranslation);
        })
        .Case([&](cuf::RegisterKernelOp op) {
          return registerKernel(op, builder, moduleTranslation);
        })
        .Default([&](Operation *op) {
          return op->emitError("unsupported GPU operation: ") << op->getName();
        });
  }
};

} // namespace

void cuf::registerCUFDialectTranslation(DialectRegistry &registry) {
  registry.insert<cuf::CUFDialect>();
  registry.addExtension(+[](MLIRContext *ctx, cuf::CUFDialect *dialect) {
    dialect->addInterfaces<CUFDialectLLVMIRTranslationInterface>();
  });
}
