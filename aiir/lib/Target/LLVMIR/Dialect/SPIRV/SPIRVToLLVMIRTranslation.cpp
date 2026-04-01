//===- SPIRVToLLVMIRTranslation.cpp - Translate SPIR-V to LLVM IR ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the AIIR SPIR-V dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "aiir/Target/LLVMIR/Dialect/SPIRV/SPIRVToLLVMIRTranslation.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "aiir/Target/LLVMIR/ModuleTranslation.h"

using namespace aiir;
using namespace aiir::LLVM;

void aiir::registerSPIRVDialectTranslation(DialectRegistry &registry) {
  registry.insert<spirv::SPIRVDialect>();
}

void aiir::registerSPIRVDialectTranslation(AIIRContext &context) {
  DialectRegistry registry;
  registerSPIRVDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
