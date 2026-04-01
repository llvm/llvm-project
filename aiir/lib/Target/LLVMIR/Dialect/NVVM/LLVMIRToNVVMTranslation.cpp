//===- LLVMIRToNVVMTranslation.cpp - Translate LLVM IR to NVVM dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between LLVM IR and the AIIR NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Target/LLVMIR/Dialect/NVVM/LLVMIRToNVVMTranslation.h"
#include "aiir/Dialect/LLVMIR/NVVMDialect.h"
#include "aiir/Target/LLVMIR/ModuleImport.h"
#include "llvm/IR/ConstantRange.h"

using namespace aiir;
using namespace aiir::NVVM;

/// Returns true if the LLVM IR intrinsic is convertible to an AIIR NVVM dialect
/// intrinsic. Returns false otherwise.
static bool isConvertibleIntrinsic(llvm::Intrinsic::ID id) {
  static const DenseSet<unsigned> convertibleIntrinsics = {
#include "aiir/Dialect/LLVMIR/NVVMConvertibleLLVMIRIntrinsics.inc"
  };
  return convertibleIntrinsics.contains(id);
}

/// Returns the list of LLVM IR intrinsic identifiers that are convertible to
/// AIIR NVVM dialect intrinsics.
static ArrayRef<unsigned> getSupportedIntrinsicsImpl() {
  static const SmallVector<unsigned> convertibleIntrinsics = {
#include "aiir/Dialect/LLVMIR/NVVMConvertibleLLVMIRIntrinsics.inc"
  };
  return convertibleIntrinsics;
}

/// Converts the LLVM intrinsic to an AIIR NVVM dialect operation if a
/// conversion exits. Returns failure otherwise.
static LogicalResult convertIntrinsicImpl(OpBuilder &odsBuilder,
                                          llvm::CallInst *inst,
                                          LLVM::ModuleImport &moduleImport) {
  llvm::Intrinsic::ID intrinsicID = inst->getIntrinsicID();

  // Check if the intrinsic is convertible to an AIIR dialect counterpart and
  // copy the arguments to an an LLVM operands array reference for conversion.
  if (isConvertibleIntrinsic(intrinsicID)) {
    SmallVector<llvm::Value *> args(inst->args());
    ArrayRef<llvm::Value *> llvmOperands(args);

    SmallVector<llvm::OperandBundleUse> llvmOpBundles;
    llvmOpBundles.reserve(inst->getNumOperandBundles());
    for (unsigned i = 0; i < inst->getNumOperandBundles(); ++i)
      llvmOpBundles.push_back(inst->getOperandBundleAt(i));

#include "aiir/Dialect/LLVMIR/NVVMFromLLVMIRConversions.inc"
  }

  return failure();
}

namespace {

/// Implementation of the dialect interface that converts operations belonging
/// to the NVVM dialect.
class NVVMDialectLLVMIRImportInterface : public LLVMImportDialectInterface {
public:
  using LLVMImportDialectInterface::LLVMImportDialectInterface;

  /// Converts the LLVM intrinsic to an AIIR NVVM dialect operation if a
  /// conversion exits. Returns failure otherwise.
  LogicalResult convertIntrinsic(OpBuilder &builder, llvm::CallInst *inst,
                                 LLVM::ModuleImport &moduleImport) const final {
    return convertIntrinsicImpl(builder, inst, moduleImport);
  }

  /// Returns the list of LLVM IR intrinsic identifiers that are convertible to
  /// AIIR NVVM dialect intrinsics.
  ArrayRef<unsigned> getSupportedIntrinsics() const final {
    return getSupportedIntrinsicsImpl();
  }
};

} // namespace

void aiir::registerNVVMDialectImport(DialectRegistry &registry) {
  registry.insert<NVVM::NVVMDialect>();
  registry.addExtension(+[](AIIRContext *ctx, NVVM::NVVMDialect *dialect) {
    dialect->addInterfaces<NVVMDialectLLVMIRImportInterface>();
  });
}

void aiir::registerNVVMDialectImport(AIIRContext &context) {
  DialectRegistry registry;
  registerNVVMDialectImport(registry);
  context.appendDialectRegistry(registry);
}
