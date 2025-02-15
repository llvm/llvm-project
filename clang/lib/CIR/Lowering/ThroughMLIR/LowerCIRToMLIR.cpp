//====- LowerCIRToMLIR.cpp - Lowering from CIR to MLIR --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR operations to MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/TimeProfiler.h"

using namespace cir;
using namespace llvm;

namespace cir {

struct ConvertCIRToMLIRPass
    : public mlir::PassWrapper<ConvertCIRToMLIRPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::memref::MemRefDialect>();
  }
  void runOnOperation() final;

  StringRef getDescription() const override {
    return "Convert the CIR dialect module to MLIR standard dialects";
  }

  StringRef getArgument() const override { return "cir-to-mlir"; }
};

class CIRGlobalOpLowering : public mlir::OpConversionPattern<cir::GlobalOp> {
public:
  using OpConversionPattern<cir::GlobalOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(cir::GlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp)
      return mlir::failure();

    mlir::OpBuilder b(moduleOp.getContext());

    const auto cirSymType = op.getSymType();
    assert(!cir::MissingFeatures::convertTypeForMemory());
    auto convertedType = getTypeConverter()->convertType(cirSymType);
    if (!convertedType)
      return mlir::failure();
    auto memrefType = dyn_cast<mlir::MemRefType>(convertedType);
    if (!memrefType)
      memrefType = mlir::MemRefType::get({}, convertedType);
    // Add an optional alignment to the global memref.
    assert(!cir::MissingFeatures::opGlobalAlignment());
    mlir::IntegerAttr memrefAlignment = mlir::IntegerAttr();
    // Add an optional initial value to the global memref.
    mlir::Attribute initialValue = mlir::Attribute();
    std::optional<mlir::Attribute> init = op.getInitialValue();
    if (init.has_value()) {
      initialValue =
          llvm::TypeSwitch<mlir::Attribute, mlir::Attribute>(init.value())
              .Case<cir::IntAttr>([&](cir::IntAttr attr) {
                auto rtt = mlir::RankedTensorType::get({}, convertedType);
                return mlir::DenseIntElementsAttr::get(rtt, attr.getValue());
              })
              .Case<cir::FPAttr>([&](cir::FPAttr attr) {
                auto rtt = mlir::RankedTensorType::get({}, convertedType);
                return mlir::DenseFPElementsAttr::get(rtt, attr.getValue());
              })
              .Default([&](mlir::Attribute attr) {
                llvm_unreachable("GlobalOp lowering with initial value is not "
                                 "fully supported yet");
                return mlir::Attribute();
              });
    }

    // Add symbol visibility
    assert(!cir::MissingFeatures::opGlobalLinkage());
    std::string symVisibility = "public";

    assert(!cir::MissingFeatures::opGlobalConstant());
    bool isConstant = false;

    rewriter.replaceOpWithNewOp<mlir::memref::GlobalOp>(
        op, b.getStringAttr(op.getSymName()),
        /*sym_visibility=*/b.getStringAttr(symVisibility),
        /*type=*/memrefType, initialValue,
        /*constant=*/isConstant,
        /*alignment=*/memrefAlignment);

    return mlir::success();
  }
};

void populateCIRToMLIRConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter) {
  patterns.add<CIRGlobalOpLowering>(converter, patterns.getContext());
}

static mlir::TypeConverter prepareTypeConverter() {
  mlir::TypeConverter converter;
  converter.addConversion([&](cir::PointerType type) -> mlir::Type {
    assert(!cir::MissingFeatures::convertTypeForMemory());
    mlir::Type ty = converter.convertType(type.getPointee());
    // FIXME: The pointee type might not be converted (e.g. struct)
    if (!ty)
      return nullptr;
    return mlir::MemRefType::get({}, ty);
  });
  converter.addConversion(
      [&](mlir::IntegerType type) -> mlir::Type { return type; });
  converter.addConversion(
      [&](mlir::FloatType type) -> mlir::Type { return type; });
  converter.addConversion([&](cir::VoidType type) -> mlir::Type { return {}; });
  converter.addConversion([&](cir::IntType type) -> mlir::Type {
    // arith dialect ops doesn't take signed integer -- drop cir sign here
    return mlir::IntegerType::get(
        type.getContext(), type.getWidth(),
        mlir::IntegerType::SignednessSemantics::Signless);
  });
  converter.addConversion([&](cir::SingleType type) -> mlir::Type {
    return mlir::Float32Type::get(type.getContext());
  });
  converter.addConversion([&](cir::DoubleType type) -> mlir::Type {
    return mlir::Float64Type::get(type.getContext());
  });
  converter.addConversion([&](cir::FP80Type type) -> mlir::Type {
    return mlir::Float80Type::get(type.getContext());
  });
  converter.addConversion([&](cir::LongDoubleType type) -> mlir::Type {
    return converter.convertType(type.getUnderlying());
  });
  converter.addConversion([&](cir::FP128Type type) -> mlir::Type {
    return mlir::Float128Type::get(type.getContext());
  });
  converter.addConversion([&](cir::FP16Type type) -> mlir::Type {
    return mlir::Float16Type::get(type.getContext());
  });
  converter.addConversion([&](cir::BF16Type type) -> mlir::Type {
    return mlir::BFloat16Type::get(type.getContext());
  });

  return converter;
}

void ConvertCIRToMLIRPass::runOnOperation() {
  auto module = getOperation();

  auto converter = prepareTypeConverter();

  mlir::RewritePatternSet patterns(&getContext());

  populateCIRToMLIRConversionPatterns(patterns, converter);

  mlir::ConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalDialect<mlir::memref::MemRefDialect>();
  target.addIllegalDialect<cir::CIRDialect>();

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> createConvertCIRToMLIRPass() {
  return std::make_unique<ConvertCIRToMLIRPass>();
}

mlir::ModuleOp lowerFromCIRToMLIR(mlir::ModuleOp mlirModule,
                                  mlir::MLIRContext &mlirCtx) {
  llvm::TimeTraceScope scope("Lower CIR To MLIR");

  mlir::PassManager pm(&mlirCtx);

  pm.addPass(createConvertCIRToMLIRPass());

  auto result = !mlir::failed(pm.run(mlirModule));
  if (!result)
    llvm::report_fatal_error(
        "The pass manager failed to lower CIR to MLIR standard dialects!");

  // Now that we ran all the lowering passes, verify the final output.
  if (mlirModule.verify().failed())
    llvm::report_fatal_error(
        "Verification of the final MLIR in standard dialects failed!");

  return mlirModule;
}

} // namespace cir
