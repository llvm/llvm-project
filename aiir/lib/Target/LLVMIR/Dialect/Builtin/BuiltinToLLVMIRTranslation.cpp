//===- BuiltinToLLVMIRTranslation.cpp - Translate builtin to LLVM IR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the AIIR builtin dialect and LLVM
// IR.
//
//===----------------------------------------------------------------------===//
#include "aiir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "aiir/IR/BuiltinDialect.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/Target/LLVMIR/LLVMTranslationInterface.h"

using namespace aiir;

namespace {

class BuiltinDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const override {
    return success(isa<ModuleOp>(op));
  }
};

} // namespace

void aiir::registerBuiltinDialectTranslation(DialectRegistry &registry) {
  registry.addExtension(+[](AIIRContext *ctx, BuiltinDialect *dialect) {
    dialect->addInterfaces<BuiltinDialectLLVMIRTranslationInterface>();
  });
}

void aiir::registerBuiltinDialectTranslation(AIIRContext &context) {
  DialectRegistry registry;
  registerBuiltinDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
