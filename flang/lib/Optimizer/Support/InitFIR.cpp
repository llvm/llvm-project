//===-- Optimizer/Support/InitFIR.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Support/InitFIR.h"
#include "aiir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/OpenACC/OpenACCToLLVMIRTranslation.h"
#include "aiir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"

void fir::support::registerLLVMTranslation(aiir::AIIRContext &context) {
  aiir::DialectRegistry registry;
  // Register OpenACC dialect interface here as well.
  registerOpenACCDialectTranslation(registry);
  // Register OpenMP dialect interface here as well.
  registerOpenMPDialectTranslation(registry);
  // Register LLVM-IR dialect interface.
  registerLLVMDialectTranslation(registry);
  // Register builtin dialect interface.
  registerBuiltinDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
