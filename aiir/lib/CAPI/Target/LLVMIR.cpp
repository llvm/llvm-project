//===-- LLVMIR.h - C Interface for AIIR LLVMIR Target ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Target/LLVMIR.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/Wrap.h"
#include "aiir/Target/LLVMIR/ModuleTranslation.h"
#include "aiir/Target/LLVMIR/TypeFromLLVM.h"

using namespace aiir;

LLVMModuleRef aiirTranslateModuleToLLVMIR(AiirOperation module,
                                          LLVMContextRef context) {
  Operation *moduleOp = unwrap(module);

  llvm::LLVMContext *ctx = llvm::unwrap(context);

  std::unique_ptr<llvm::Module> llvmModule =
      aiir::translateModuleToLLVMIR(moduleOp, *ctx);

  LLVMModuleRef moduleRef = llvm::wrap(llvmModule.release());

  return moduleRef;
}

char *aiirTranslateModuleToLLVMIRToString(AiirOperation module) {
  LLVMContextRef llvmCtx = LLVMContextCreate();
  LLVMModuleRef llvmModule = aiirTranslateModuleToLLVMIR(module, llvmCtx);
  char *llvmir = LLVMPrintModuleToString(llvmModule);
  LLVMDisposeModule(llvmModule);
  LLVMContextDispose(llvmCtx);
  return llvmir;
}

DEFINE_C_API_PTR_METHODS(AiirTypeFromLLVMIRTranslator,
                         aiir::LLVM::TypeFromLLVMIRTranslator)

AiirTypeFromLLVMIRTranslator
aiirTypeFromLLVMIRTranslatorCreate(AiirContext ctx) {
  AIIRContext *context = unwrap(ctx);
  auto *translator = new LLVM::TypeFromLLVMIRTranslator(*context);
  return wrap(translator);
}

void aiirTypeFromLLVMIRTranslatorDestroy(
    AiirTypeFromLLVMIRTranslator translator) {
  delete static_cast<LLVM::TypeFromLLVMIRTranslator *>(unwrap(translator));
}

AiirType aiirTypeFromLLVMIRTranslatorTranslateType(
    AiirTypeFromLLVMIRTranslator translator, LLVMTypeRef llvmType) {
  LLVM::TypeFromLLVMIRTranslator *translator_ = unwrap(translator);
  aiir::Type type = translator_->translateType(llvm::unwrap(llvmType));
  return wrap(type);
}

DEFINE_C_API_PTR_METHODS(AiirTypeToLLVMIRTranslator,
                         aiir::LLVM::TypeToLLVMIRTranslator)

AiirTypeToLLVMIRTranslator
aiirTypeToLLVMIRTranslatorCreate(LLVMContextRef ctx) {
  llvm::LLVMContext *context = llvm::unwrap(ctx);
  auto *translator = new LLVM::TypeToLLVMIRTranslator(*context);
  return wrap(translator);
}

void aiirTypeToLLVMIRTranslatorDestroy(AiirTypeToLLVMIRTranslator translator) {
  delete static_cast<LLVM::TypeToLLVMIRTranslator *>(unwrap(translator));
}

LLVMTypeRef
aiirTypeToLLVMIRTranslatorTranslateType(AiirTypeToLLVMIRTranslator translator,
                                        AiirType aiirType) {
  LLVM::TypeToLLVMIRTranslator *translator_ = unwrap(translator);
  llvm::Type *type = translator_->translateType(unwrap(aiirType));
  return llvm::wrap(type);
}
