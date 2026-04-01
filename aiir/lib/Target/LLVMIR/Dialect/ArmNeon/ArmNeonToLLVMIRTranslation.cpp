//===- ArmNeonToLLVMIRTranslation.cpp - Translate ArmNeon to LLVM IR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the AIIR ArmNeon dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "aiir/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.h"
#include "aiir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "aiir/IR/Operation.h"
#include "aiir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAArch64.h"

using namespace aiir;
using namespace aiir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the ArmNeon dialect to LLVM IR.
class ArmNeonDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "aiir/Dialect/ArmNeon/ArmNeonConversions.inc"

    return failure();
  }
};
} // namespace

void aiir::registerArmNeonDialectTranslation(DialectRegistry &registry) {
  registry.insert<arm_neon::ArmNeonDialect>();
  registry.addExtension(
      +[](AIIRContext *ctx, arm_neon::ArmNeonDialect *dialect) {
        dialect->addInterfaces<ArmNeonDialectLLVMIRTranslationInterface>();
      });
}

void aiir::registerArmNeonDialectTranslation(AIIRContext &context) {
  DialectRegistry registry;
  registerArmNeonDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
