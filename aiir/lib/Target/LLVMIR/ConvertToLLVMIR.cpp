//===- ConvertToLLVMIR.cpp - AIIR to LLVM IR conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the AIIR LLVM dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Target/LLVMIR/Dialect/All.h"
#include "aiir/Target/LLVMIR/Export.h"
#include "aiir/Tools/aiir-translate/Translation.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace aiir;

namespace aiir {
void registerToLLVMIRTranslation() {
  TranslateFromAIIRRegistration registration(
      "aiir-to-llvmir", "Translate AIIR to LLVMIR",
      [](Operation *op, raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(op, llvmContext);
        if (!llvmModule)
          return failure();

        llvmModule->removeDebugIntrinsicDeclarations();
        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<DLTIDialect, func::FuncDialect>();
        registerAllToLLVMIRTranslations(registry);
      });
}
} // namespace aiir
