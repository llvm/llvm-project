//===- RegisterEverything.cpp - Register all AIIR entities ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/RegisterEverything.h"

#include "aiir/CAPI/IR.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/InitAllDialects.h"
#include "aiir/InitAllExtensions.h"
#include "aiir/InitAllPasses.h"
#include "aiir/Target/LLVMIR/Dialect/All.h"
#include "aiir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

void aiirRegisterAllDialects(AiirDialectRegistry registry) {
  aiir::registerAllDialects(*unwrap(registry));
  aiir::registerAllExtensions(*unwrap(registry));
}

void aiirRegisterAllLLVMTranslations(AiirContext context) {
  auto &ctx = *unwrap(context);
  aiir::DialectRegistry registry;
  aiir::registerAllToLLVMIRTranslations(registry);
  ctx.appendDialectRegistry(registry);
}

void aiirRegisterAllPasses() { aiir::registerAllPasses(); }
