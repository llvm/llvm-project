//==- CXXABILowering.cpp - lower C++ operations to target-specific ABI form -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "TargetLowering/LowerModule.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/MissingFeatures.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CXXABILOWERING
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

#define GET_ABI_LOWERING_PATTERNS
#include "clang/CIR/Dialect/IR/CIRLowering.inc"
#undef GET_ABI_LOWERING_PATTERNS

struct CXXABILoweringPass
    : public impl::CXXABILoweringBase<CXXABILoweringPass> {
  CXXABILoweringPass() = default;
  void runOnOperation() override;
};

/// A generic ABI lowering rewrite pattern. This conversion pattern matches any
/// CIR dialect operations with at least one operand or result of an
/// ABI-dependent type. This conversion pattern rewrites the matched operation
/// by replacing all its ABI-dependent operands and results with their
/// lowered counterparts.
class CIRGenericCXXABILoweringPattern : public mlir::ConversionPattern {
public:
  CIRGenericCXXABILoweringPattern(mlir::MLIRContext *context,
                                  const mlir::TypeConverter &typeConverter)
      : mlir::ConversionPattern(typeConverter, MatchAnyOpTypeTag(),
                                /*benefit=*/1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Do not match on operations that have dedicated ABI lowering rewrite rules
    if (llvm::isa<cir::AllocaOp, cir::BaseDataMemberOp, cir::CastOp, cir::CmpOp,
                  cir::ConstantOp, cir::DerivedDataMemberOp, cir::FuncOp,
                  cir::GetMethodOp, cir::GetRuntimeMemberOp, cir::GlobalOp>(op))
      return mlir::failure();

    const mlir::TypeConverter *typeConverter = getTypeConverter();
    assert(typeConverter &&
           "CIRGenericCXXABILoweringPattern requires a type converter");
    bool operandsAndResultsLegal = typeConverter->isLegal(op);
    bool regionsLegal =
        std::all_of(op->getRegions().begin(), op->getRegions().end(),
                    [typeConverter](mlir::Region &region) {
                      return typeConverter->isLegal(&region);
                    });
    if (operandsAndResultsLegal && regionsLegal) {
      // The operation does not have any CXXABI-dependent operands or results,
      // the match fails.
      return mlir::failure();
    }

    assert(op->getNumRegions() == 0 && "CIRGenericCXXABILoweringPattern cannot "
                                       "deal with operations with regions");

    mlir::OperationState loweredOpState(op->getLoc(), op->getName());
    loweredOpState.addOperands(operands);
    loweredOpState.addAttributes(op->getAttrs());
    loweredOpState.addSuccessors(op->getSuccessors());

    // Lower all result types
    llvm::SmallVector<mlir::Type> loweredResultTypes;
    loweredResultTypes.reserve(op->getNumResults());
    for (mlir::Type result : op->getResultTypes())
      loweredResultTypes.push_back(typeConverter->convertType(result));
    loweredOpState.addTypes(loweredResultTypes);

    // Lower all regions
    for (mlir::Region &region : op->getRegions()) {
      mlir::Region *loweredRegion = loweredOpState.addRegion();
      rewriter.inlineRegionBefore(region, *loweredRegion, loweredRegion->end());
      if (mlir::failed(
              rewriter.convertRegionTypes(loweredRegion, *getTypeConverter())))
        return mlir::failure();
    }

    // Clone the operation with lowered operand types and result types
    mlir::Operation *loweredOp = rewriter.create(loweredOpState);

    rewriter.replaceOp(op, loweredOp);
    return mlir::success();
  }
};

} // namespace

mlir::LogicalResult CIRAllocaOpABILowering::matchAndRewrite(
    cir::AllocaOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type allocaPtrTy = op.getType();
  mlir::Type allocaTy = op.getAllocaType();
  mlir::Type loweredAllocaPtrTy = getTypeConverter()->convertType(allocaPtrTy);
  mlir::Type loweredAllocaTy = getTypeConverter()->convertType(allocaTy);

  cir::AllocaOp loweredOp = cir::AllocaOp::create(
      rewriter, op.getLoc(), loweredAllocaPtrTy, loweredAllocaTy, op.getName(),
      op.getAlignmentAttr(), /*dynAllocSize=*/adaptor.getDynAllocSize());
  loweredOp.setInit(op.getInit());
  loweredOp.setConstant(op.getConstant());
  loweredOp.setAnnotationsAttr(op.getAnnotationsAttr());

  rewriter.replaceOp(op, loweredOp);
  return mlir::success();
}

mlir::LogicalResult CIRCastOpABILowering::matchAndRewrite(
    cir::CastOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type srcTy = op.getSrc().getType();
  assert((mlir::isa<cir::DataMemberType, cir::MethodType>(srcTy)) &&
         "input to bitcast in ABI lowering must be a data member or method");

  switch (op.getKind()) {
  case cir::CastKind::bitcast: {
    mlir::Type destTy = getTypeConverter()->convertType(op.getType());
    mlir::Value loweredResult;
    if (mlir::isa<cir::DataMemberType>(srcTy))
      loweredResult = lowerModule->getCXXABI().lowerDataMemberBitcast(
          op, destTy, adaptor.getSrc(), rewriter);
    else
      loweredResult = lowerModule->getCXXABI().lowerMethodBitcast(
          op, destTy, adaptor.getSrc(), rewriter);
    rewriter.replaceOp(op, loweredResult);
    return mlir::success();
  }
  case cir::CastKind::member_ptr_to_bool: {
    mlir::Value loweredResult;
    if (mlir::isa<cir::MethodType>(srcTy))
      loweredResult = lowerModule->getCXXABI().lowerMethodToBoolCast(
          op, adaptor.getSrc(), rewriter);
    else
      loweredResult = lowerModule->getCXXABI().lowerDataMemberToBoolCast(
          op, adaptor.getSrc(), rewriter);
    rewriter.replaceOp(op, loweredResult);
    return mlir::success();
  }
  default:
    break;
  }

  return mlir::failure();
}

mlir::LogicalResult CIRConstantOpABILowering::matchAndRewrite(
    cir::ConstantOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  if (mlir::isa<cir::DataMemberType>(op.getType())) {
    auto dataMember = mlir::cast<cir::DataMemberAttr>(op.getValue());
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    mlir::TypedAttr abiValue = lowerModule->getCXXABI().lowerDataMemberConstant(
        dataMember, layout, *getTypeConverter());
    rewriter.replaceOpWithNewOp<ConstantOp>(op, abiValue);
    return mlir::success();
  }

  if (mlir::isa<cir::MethodType>(op.getType())) {
    auto method = mlir::cast<cir::MethodAttr>(op.getValue());
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    mlir::TypedAttr abiValue = lowerModule->getCXXABI().lowerMethodConstant(
        method, layout, *getTypeConverter());
    rewriter.replaceOpWithNewOp<ConstantOp>(op, abiValue);
    return mlir::success();
  }

  llvm_unreachable("constant operand is not an CXXABI-dependent type");
}

mlir::LogicalResult CIRCmpOpABILowering::matchAndRewrite(
    cir::CmpOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type type = op.getLhs().getType();
  assert((mlir::isa<cir::DataMemberType, cir::MethodType>(type)) &&
         "input to cmp in ABI lowering must be a data member or method");

  mlir::Value loweredResult;
  if (mlir::isa<cir::DataMemberType>(type))
    loweredResult = lowerModule->getCXXABI().lowerDataMemberCmp(
        op, adaptor.getLhs(), adaptor.getRhs(), rewriter);
  else
    loweredResult = lowerModule->getCXXABI().lowerMethodCmp(
        op, adaptor.getLhs(), adaptor.getRhs(), rewriter);

  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

mlir::LogicalResult CIRFuncOpABILowering::matchAndRewrite(
    cir::FuncOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  cir::FuncType opFuncType = op.getFunctionType();
  mlir::TypeConverter::SignatureConversion signatureConversion(
      opFuncType.getNumInputs());

  for (const auto &[i, argType] : llvm::enumerate(opFuncType.getInputs())) {
    mlir::Type loweredArgType = getTypeConverter()->convertType(argType);
    if (!loweredArgType)
      return mlir::failure();
    signatureConversion.addInputs(i, loweredArgType);
  }

  mlir::Type loweredResultType =
      getTypeConverter()->convertType(opFuncType.getReturnType());
  if (!loweredResultType)
    return mlir::failure();

  auto loweredFuncType =
      cir::FuncType::get(signatureConversion.getConvertedTypes(),
                         loweredResultType, /*isVarArg=*/opFuncType.isVarArg());

  // Create a new cir.func operation for the CXXABI-lowered function.
  cir::FuncOp loweredFuncOp = rewriter.cloneWithoutRegions(op);
  loweredFuncOp.setFunctionType(loweredFuncType);
  rewriter.inlineRegionBefore(op.getBody(), loweredFuncOp.getBody(),
                              loweredFuncOp.end());
  if (mlir::failed(rewriter.convertRegionTypes(
          &loweredFuncOp.getBody(), *getTypeConverter(), &signatureConversion)))
    return mlir::failure();

  rewriter.eraseOp(op);
  return mlir::success();
}

mlir::LogicalResult CIRGlobalOpABILowering::matchAndRewrite(
    cir::GlobalOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type ty = op.getSymType();
  mlir::Type loweredTy = getTypeConverter()->convertType(ty);
  if (!loweredTy)
    return mlir::failure();

  mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());

  mlir::Attribute loweredInit;
  if (mlir::isa<cir::DataMemberType>(ty)) {
    cir::DataMemberAttr init =
        mlir::cast_if_present<cir::DataMemberAttr>(op.getInitialValueAttr());
    loweredInit = lowerModule->getCXXABI().lowerDataMemberConstant(
        init, layout, *getTypeConverter());
  } else {
    llvm_unreachable(
        "inputs to cir.global in ABI lowering must be data member or method");
  }

  auto newOp = mlir::cast<cir::GlobalOp>(rewriter.clone(*op.getOperation()));
  newOp.setInitialValueAttr(loweredInit);
  newOp.setSymType(loweredTy);
  rewriter.replaceOp(op, newOp);
  return mlir::success();
}

mlir::LogicalResult CIRBaseDataMemberOpABILowering::matchAndRewrite(
    cir::BaseDataMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult = lowerModule->getCXXABI().lowerBaseDataMember(
      op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

mlir::LogicalResult CIRDerivedDataMemberOpABILowering::matchAndRewrite(
    cir::DerivedDataMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult = lowerModule->getCXXABI().lowerDerivedDataMember(
      op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

mlir::LogicalResult CIRDynamicCastOpABILowering::matchAndRewrite(
    cir::DynamicCastOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult =
      lowerModule->getCXXABI().lowerDynamicCast(op, rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

mlir::LogicalResult CIRGetMethodOpABILowering::matchAndRewrite(
    cir::GetMethodOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value callee;
  mlir::Value thisArg;
  lowerModule->getCXXABI().lowerGetMethod(
      op, callee, thisArg, adaptor.getMethod(), adaptor.getObject(), rewriter);
  rewriter.replaceOp(op, {callee, thisArg});
  return mlir::success();
}

mlir::LogicalResult CIRGetRuntimeMemberOpABILowering::matchAndRewrite(
    cir::GetRuntimeMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type resTy = getTypeConverter()->convertType(op.getType());
  mlir::Operation *newOp = lowerModule->getCXXABI().lowerGetRuntimeMember(
      op, resTy, adaptor.getAddr(), adaptor.getMember(), rewriter);
  rewriter.replaceOp(op, newOp);
  return mlir::success();
}

// Prepare the type converter for the CXXABI lowering pass.
// Even though this is a CIR-to-CIR pass, we are eliminating some CIR types.
static void prepareCXXABITypeConverter(mlir::TypeConverter &converter,
                                       mlir::DataLayout &dataLayout,
                                       cir::LowerModule &lowerModule) {
  converter.addConversion([&](mlir::Type type) -> mlir::Type { return type; });
  // This is necessary in order to convert CIR pointer types that are pointing
  // to CIR types that we are lowering in this pass.
  converter.addConversion([&](cir::PointerType type) -> mlir::Type {
    mlir::Type loweredPointeeType = converter.convertType(type.getPointee());
    if (!loweredPointeeType)
      return {};
    return cir::PointerType::get(type.getContext(), loweredPointeeType,
                                 type.getAddrSpace());
  });
  converter.addConversion([&](cir::DataMemberType type) -> mlir::Type {
    mlir::Type abiType =
        lowerModule.getCXXABI().lowerDataMemberType(type, converter);
    return converter.convertType(abiType);
  });
  converter.addConversion([&](cir::MethodType type) -> mlir::Type {
    mlir::Type abiType =
        lowerModule.getCXXABI().lowerMethodType(type, converter);
    return converter.convertType(abiType);
  });
  // This is necessary in order to convert CIR function types that have argument
  // or return types that use CIR types that we are lowering in this pass.
  converter.addConversion([&](cir::FuncType type) -> mlir::Type {
    llvm::SmallVector<mlir::Type> loweredInputTypes;
    loweredInputTypes.reserve(type.getNumInputs());
    if (mlir::failed(
            converter.convertTypes(type.getInputs(), loweredInputTypes)))
      return {};

    mlir::Type loweredReturnType = converter.convertType(type.getReturnType());
    if (!loweredReturnType)
      return {};

    return cir::FuncType::get(loweredInputTypes, loweredReturnType,
                              /*isVarArg=*/type.getVarArg());
  });
}

static void
populateCXXABIConversionTarget(mlir::ConversionTarget &target,
                               const mlir::TypeConverter &typeConverter) {
  target.addLegalOp<mlir::ModuleOp>();

  // The ABI lowering pass is interested in CIR operations with operands or
  // results of CXXABI-dependent types, or CIR operations with regions whose
  // block arguments are of CXXABI-dependent types.
  target.addDynamicallyLegalDialect<cir::CIRDialect>(
      [&typeConverter](mlir::Operation *op) {
        if (!typeConverter.isLegal(op))
          return false;
        return std::all_of(op->getRegions().begin(), op->getRegions().end(),
                           [&typeConverter](mlir::Region &region) {
                             return typeConverter.isLegal(&region);
                           });
      });

  // Some CIR ops needs special checking for legality
  target.addDynamicallyLegalOp<cir::FuncOp>([&typeConverter](cir::FuncOp op) {
    return typeConverter.isLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<cir::GlobalOp>(
      [&typeConverter](cir::GlobalOp op) {
        return typeConverter.isLegal(op.getSymType());
      });
  target.addIllegalOp<cir::DynamicCastOp>();
}

//===----------------------------------------------------------------------===//
// The Pass
//===----------------------------------------------------------------------===//

void CXXABILoweringPass::runOnOperation() {
  auto module = mlir::cast<mlir::ModuleOp>(getOperation());
  mlir::MLIRContext *ctx = module.getContext();

  // If the triple is not present, e.g. CIR modules parsed from text, we
  // cannot init LowerModule properly.
  assert(!cir::MissingFeatures::makeTripleAlwaysPresent());
  // If no target triple is available, skip the ABI lowering pass.
  if (!module->hasAttr(cir::CIRDialect::getTripleAttrName()))
    return;

  mlir::PatternRewriter rewriter(ctx);
  std::unique_ptr<cir::LowerModule> lowerModule =
      cir::createLowerModule(module, rewriter);

  mlir::DataLayout dataLayout(module);
  mlir::TypeConverter typeConverter;
  prepareCXXABITypeConverter(typeConverter, dataLayout, *lowerModule);

  mlir::RewritePatternSet patterns(ctx);
  patterns.add<CIRGenericCXXABILoweringPattern>(patterns.getContext(),
                                                typeConverter);
  patterns.add<
#define GET_ABI_LOWERING_PATTERNS_LIST
#include "clang/CIR/Dialect/IR/CIRLowering.inc"
#undef GET_ABI_LOWERING_PATTERNS_LIST
      >(patterns.getContext(), typeConverter, dataLayout, *lowerModule);

  mlir::ConversionTarget target(*ctx);
  populateCXXABIConversionTarget(target, typeConverter);

  if (failed(mlir::applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createCXXABILoweringPass() {
  return std::make_unique<CXXABILoweringPass>();
}
