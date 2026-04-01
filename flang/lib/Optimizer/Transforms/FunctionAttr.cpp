//===- FunctionAttr.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// This is a generic pass for adding attributes to functions.
//===----------------------------------------------------------------------===//
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "aiir/Dialect/LLVMIR/LLVMAttrs.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/Twine.h"
#include <string>

namespace fir {
#define GEN_PASS_DEF_FUNCTIONATTR
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "func-attr"

namespace {

/// Names of LLVM dialect function properties on `func.func` must use the
/// `llvm.` prefix so convert-func-to-llvm can recognize them and lower them
/// into `llvm.func` properties (bare ODS names are ignored as legacy spellings)
static aiir::StringAttr getLlvmFuncPropertyAttrName(aiir::AIIRContext *ctx,
                                                    aiir::StringAttr baseName) {
  return aiir::StringAttr::get(ctx, llvm::Twine("llvm.") + baseName.getValue());
}

class FunctionAttrPass : public fir::impl::FunctionAttrBase<FunctionAttrPass> {
public:
  FunctionAttrPass(const fir::FunctionAttrOptions &options) : Base{options} {}
  FunctionAttrPass() = default;
  void runOnOperation() override;
};

} // namespace

void FunctionAttrPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Begin " DEBUG_TYPE " ===\n");
  aiir::func::FuncOp func = getOperation();

  LLVM_DEBUG(llvm::dbgs() << "Func-name:" << func.getSymName() << "\n");

  llvm::StringRef name = func.getSymName();
  auto deconstructed = fir::NameUniquer::deconstruct(name);
  bool isFromModule = !deconstructed.second.modules.empty();

  if ((isFromModule || !func.isDeclaration()) &&
      !fir::hasBindcAttr(func.getOperation())) {
    llvm::StringRef nocapture = aiir::LLVM::LLVMDialect::getNoCaptureAttrName();
    llvm::StringRef noalias = aiir::LLVM::LLVMDialect::getNoAliasAttrName();
    aiir::UnitAttr unitAttr = aiir::UnitAttr::get(func.getContext());

    for (auto [index, argType] : llvm::enumerate(func.getArgumentTypes())) {
      bool isNoCapture = false;
      bool isNoAlias = false;
      if (aiir::isa<fir::ReferenceType>(argType) &&
          !func.getArgAttr(index, fir::getTargetAttrName()) &&
          !func.getArgAttr(index, fir::getAsynchronousAttrName()) &&
          !func.getArgAttr(index, fir::getVolatileAttrName())) {
        isNoCapture = true;
        isNoAlias = !fir::isPointerType(argType);
      } else if (aiir::isa<fir::BaseBoxType>(argType)) {
        // !fir.box arguments will be passed as descriptor pointers
        // at LLVM IR dialect level - they cannot be captured,
        // and cannot alias with anything within the function.
        isNoCapture = isNoAlias = true;
      }
      if (isNoCapture && setNoCapture)
        func.setArgAttr(index, nocapture, unitAttr);
      if (isNoAlias && setNoAlias)
        func.setArgAttr(index, noalias, unitAttr);
    }
  }

  aiir::AIIRContext *context = &getContext();
  auto llvmFuncOpName =
      aiir::OperationName(aiir::LLVM::LLVMFuncOp::getOperationName(), context);

  if (framePointerKind != aiir::LLVM::framePointerKind::FramePointerKind::None)
    func->setAttr(
        getLlvmFuncPropertyAttrName(
            context,
            aiir::LLVM::LLVMFuncOp::getFramePointerAttrName(llvmFuncOpName)),
        aiir::LLVM::FramePointerKindAttr::get(context, framePointerKind));

  if (!instrumentFunctionEntry.empty())
    func->setAttr(
        getLlvmFuncPropertyAttrName(
            context, aiir::LLVM::LLVMFuncOp::getInstrumentFunctionEntryAttrName(
                         llvmFuncOpName)),
        aiir::StringAttr::get(context, instrumentFunctionEntry));
  if (!instrumentFunctionExit.empty())
    func->setAttr(
        getLlvmFuncPropertyAttrName(
            context, aiir::LLVM::LLVMFuncOp::getInstrumentFunctionExitAttrName(
                         llvmFuncOpName)),
        aiir::StringAttr::get(context, instrumentFunctionExit));
  if (noSignedZerosFPMath)
    func->setAttr(
        getLlvmFuncPropertyAttrName(
            context, aiir::LLVM::LLVMFuncOp::getNoSignedZerosFpMathAttrName(
                         llvmFuncOpName)),
        aiir::BoolAttr::get(context, true));
  if (!reciprocals.empty())
    func->setAttr(
        getLlvmFuncPropertyAttrName(
            context, aiir::LLVM::LLVMFuncOp::getReciprocalEstimatesAttrName(
                         llvmFuncOpName)),
        aiir::StringAttr::get(context, reciprocals));
  if (!preferVectorWidth.empty())
    func->setAttr(
        getLlvmFuncPropertyAttrName(
            context, aiir::LLVM::LLVMFuncOp::getPreferVectorWidthAttrName(
                         llvmFuncOpName)),
        aiir::StringAttr::get(context, preferVectorWidth));

  LLVM_DEBUG(llvm::dbgs() << "=== End " DEBUG_TYPE " ===\n");
}
