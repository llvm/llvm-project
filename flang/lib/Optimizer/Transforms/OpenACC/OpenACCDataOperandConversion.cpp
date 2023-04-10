//===- OpenACCDataOperandConversion.cpp - OpenACC data operand conversion -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/OpenACCToLLVM/ConvertOpenACCToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace fir {
#define GEN_PASS_DEF_OPENACCDATAOPERANDCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-openacc-conversion"
#include "../CodeGen/TypeConverter.h"

using namespace fir;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

template <typename Op>
class LegalizeDataOpForLLVMTranslation : public ConvertOpToLLVMPattern<Op> {
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &builder) const override {
    Location loc = op.getLoc();
    fir::LLVMTypeConverter &converter =
        *static_cast<fir::LLVMTypeConverter *>(this->getTypeConverter());

    unsigned numDataOperands = op.getNumDataOperands();

    // Keep the non data operands without modification.
    auto nonDataOperands = adaptor.getOperands().take_front(
        adaptor.getOperands().size() - numDataOperands);
    SmallVector<Value> convertedOperands;
    convertedOperands.append(nonDataOperands.begin(), nonDataOperands.end());

    // Go over the data operand and legalize them for translation.
    for (unsigned idx = 0; idx < numDataOperands; ++idx) {
      Value originalDataOperand = op.getDataOperand(idx);
      if (auto refTy =
              originalDataOperand.getType().dyn_cast<fir::ReferenceType>()) {
        if (refTy.getEleTy().isa<fir::BaseBoxType>())
          return builder.notifyMatchFailure(op, "BaseBoxType not supported");
        mlir::Type convertedType =
            converter.convertType(refTy).cast<mlir::LLVM::LLVMPointerType>();
        mlir::Value castedOperand =
            builder
                .create<mlir::UnrealizedConversionCastOp>(loc, convertedType,
                                                          originalDataOperand)
                .getResult(0);
        convertedOperands.push_back(castedOperand);
      } else {
        // Type not supported.
        return builder.notifyMatchFailure(op, "expecting a reference type");
      }
    }

    builder.replaceOpWithNewOp<Op>(op, TypeRange(), convertedOperands,
                                   op.getOperation()->getAttrs());

    return success();
  }
};
} // namespace

namespace {
struct OpenACCDataOperandConversion
    : public fir::impl::OpenACCDataOperandConversionBase<
          OpenACCDataOperandConversion> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void OpenACCDataOperandConversion::runOnOperation() {
  auto op = getOperation();
  auto *context = op.getContext();

  // Convert to OpenACC operations with LLVM IR dialect
  RewritePatternSet patterns(context);
  LowerToLLVMOptions options(context);
  options.useOpaquePointers = useOpaquePointers;
  fir::LLVMTypeConverter converter(
      op.getOperation()->getParentOfType<mlir::ModuleOp>(), true);
  patterns.add<LegalizeDataOpForLLVMTranslation<acc::DataOp>>(converter);
  patterns.add<LegalizeDataOpForLLVMTranslation<acc::EnterDataOp>>(converter);
  patterns.add<LegalizeDataOpForLLVMTranslation<acc::ExitDataOp>>(converter);
  patterns.add<LegalizeDataOpForLLVMTranslation<acc::ParallelOp>>(converter);
  patterns.add<LegalizeDataOpForLLVMTranslation<acc::UpdateOp>>(converter);

  ConversionTarget target(*context);
  target.addLegalDialect<fir::FIROpsDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  auto allDataOperandsAreConverted = [](ValueRange operands) {
    for (Value operand : operands) {
      if (!operand.getType().isa<LLVM::LLVMPointerType>())
        return false;
    }
    return true;
  };

  target.addDynamicallyLegalOp<acc::DataOp>(
      [allDataOperandsAreConverted](acc::DataOp op) {
        return allDataOperandsAreConverted(op.getCopyOperands()) &&
               allDataOperandsAreConverted(op.getCopyinOperands()) &&
               allDataOperandsAreConverted(op.getCopyinReadonlyOperands()) &&
               allDataOperandsAreConverted(op.getCopyoutOperands()) &&
               allDataOperandsAreConverted(op.getCopyoutZeroOperands()) &&
               allDataOperandsAreConverted(op.getCreateOperands()) &&
               allDataOperandsAreConverted(op.getCreateZeroOperands()) &&
               allDataOperandsAreConverted(op.getNoCreateOperands()) &&
               allDataOperandsAreConverted(op.getPresentOperands()) &&
               allDataOperandsAreConverted(op.getDeviceptrOperands()) &&
               allDataOperandsAreConverted(op.getAttachOperands());
      });

  target.addDynamicallyLegalOp<acc::EnterDataOp>(
      [allDataOperandsAreConverted](acc::EnterDataOp op) {
        return allDataOperandsAreConverted(op.getCopyinOperands()) &&
               allDataOperandsAreConverted(op.getCreateOperands()) &&
               allDataOperandsAreConverted(op.getCreateZeroOperands()) &&
               allDataOperandsAreConverted(op.getAttachOperands());
      });

  target.addDynamicallyLegalOp<acc::ExitDataOp>(
      [allDataOperandsAreConverted](acc::ExitDataOp op) {
        return allDataOperandsAreConverted(op.getCopyoutOperands()) &&
               allDataOperandsAreConverted(op.getDeleteOperands()) &&
               allDataOperandsAreConverted(op.getDetachOperands());
      });

  target.addDynamicallyLegalOp<acc::ParallelOp>(
      [allDataOperandsAreConverted](acc::ParallelOp op) {
        return allDataOperandsAreConverted(op.getReductionOperands()) &&
               allDataOperandsAreConverted(op.getCopyOperands()) &&
               allDataOperandsAreConverted(op.getCopyinOperands()) &&
               allDataOperandsAreConverted(op.getCopyinReadonlyOperands()) &&
               allDataOperandsAreConverted(op.getCopyoutOperands()) &&
               allDataOperandsAreConverted(op.getCopyoutZeroOperands()) &&
               allDataOperandsAreConverted(op.getCreateOperands()) &&
               allDataOperandsAreConverted(op.getCreateZeroOperands()) &&
               allDataOperandsAreConverted(op.getNoCreateOperands()) &&
               allDataOperandsAreConverted(op.getPresentOperands()) &&
               allDataOperandsAreConverted(op.getDevicePtrOperands()) &&
               allDataOperandsAreConverted(op.getAttachOperands()) &&
               allDataOperandsAreConverted(op.getGangPrivateOperands()) &&
               allDataOperandsAreConverted(op.getGangFirstPrivateOperands());
      });

  target.addDynamicallyLegalOp<acc::UpdateOp>(
      [allDataOperandsAreConverted](acc::UpdateOp op) {
        return allDataOperandsAreConverted(op.getHostOperands()) &&
               allDataOperandsAreConverted(op.getDeviceOperands());
      });

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}
