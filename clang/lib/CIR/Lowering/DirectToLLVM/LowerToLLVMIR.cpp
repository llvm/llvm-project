//===----------------------------------------------------------------------===//
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
#include "clang/CIR/MissingFeatures.h"
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
    if (auto func = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(op)) {
      amendFunction(func, instructions, attribute, moduleTranslation);
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

  // Translate CIR's inline attribute to LLVM's function attributes.
  void amendFunction(mlir::LLVM::LLVMFuncOp func,
                     llvm::ArrayRef<llvm::Instruction *> instructions,
                     mlir::NamedAttribute attribute,
                     mlir::LLVM::ModuleTranslation &moduleTranslation) const {
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());
    if (auto inlineAttr = mlir::dyn_cast<cir::InlineAttr>(attribute.getValue())) {
      if (inlineAttr.isNoInline())
        llvmFunc->addFnAttr(llvm::Attribute::NoInline);
      else if (inlineAttr.isAlwaysInline())
        llvmFunc->addFnAttr(llvm::Attribute::AlwaysInline);
      else if (inlineAttr.isInlineHint())
        llvmFunc->addFnAttr(llvm::Attribute::InlineHint);
      else
        llvm_unreachable("Unknown inline kind");
      // Drop ammended CIR attribute from LLVM op.
      func->removeAttr(attribute.getName());
    }

    assert(!cir::MissingFeatures::opFuncOptNoneAttr());
    assert(!cir::MissingFeatures::opFuncNoUnwind());
    assert(!cir::MissingFeatures::opFuncColdHotAttr());
    assert(!cir::MissingFeatures::opFuncUnwindTablesAttr());
    assert(!cir::MissingFeatures::openCL());
  }
};

void registerCIRDialectTranslation(mlir::DialectRegistry &registry) {
  registry.insert<cir::CIRDialect>();
  registry.addExtension(+[](mlir::MLIRContext *ctx, cir::CIRDialect *dialect) {
    dialect->addInterfaces<CIRDialectLLVMIRTranslationInterface>();
  });
}

} // namespace direct
} // namespace cir

namespace mlir {
void registerCIRDialectTranslation(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  cir::direct::registerCIRDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
} // namespace mlir
