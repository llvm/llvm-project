//==- CXXABILowering.cpp - lower C++ operations to target-specific ABI form -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <deque>

#include "PassDetail.h"
#include "TargetLowering/LowerModule.h"

#include "mlir/Dialect/OpenACC/OpenACCOpsDialect.h.inc"
#include "mlir/Dialect/OpenMP/OpenMPOpsDialect.h.inc"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/MissingFeatures.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CXXABILOWERING
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {
// Check an attribute for legality. An attribute is only currently potentially
// illegal if it contains a type, member pointers are our source of illegality
// in regards to attributes.
bool isCXXABIAttributeLegal(const mlir::TypeConverter &tc,
                            mlir::Attribute attr) {
  // If we don't have an attribute, it can't have a type!
  if (!attr)
    return true;

  // None of the OpenACC/OMP attributes contain a type of concern, so we can
  // just treat them as legal.
  if (isa<mlir::acc::OpenACCDialect, mlir::omp::OpenMPDialect>(
          attr.getDialect()))
    return true;

  // These attributes either don't contain a type, or don't contain a type that
  // can have a data member/method.
  if (isa<mlir::DenseArrayAttr, mlir::FloatAttr, mlir::UnitAttr,
          mlir::StringAttr, mlir::IntegerAttr, mlir::SymbolRefAttr,
          cir::AnnotationAttr>(attr))
    return true;

  // Tablegen'ed always-legal attributes:
  if (isa<
#define CXX_ABI_ALWAYS_LEGAL_ATTRS
#include "clang/CIR/Dialect/IR/CIRLowering.inc"
#undef CXX_ABI_ALWAYS_LEGAL_ATTRS
          >(attr))
    return true;

  // Data Member and method are ALWAYS illegal.
  if (isa<cir::DataMemberAttr, cir::MethodAttr>(attr))
    return false;

  return llvm::TypeSwitch<mlir::Attribute, bool>(attr)
      // These attributes just have a type, so they are legal if their type is.
      .Case<cir::ZeroAttr>(
          [&tc](cir::ZeroAttr za) { return tc.isLegal(za.getType()); })
      .Case<cir::PoisonAttr>(
          [&tc](cir::PoisonAttr pa) { return tc.isLegal(pa.getType()); })
      .Case<cir::UndefAttr>(
          [&tc](cir::UndefAttr uda) { return tc.isLegal(uda.getType()); })
      .Case<mlir::TypeAttr>(
          [&tc](mlir::TypeAttr ta) { return tc.isLegal(ta.getValue()); })
      .Case<cir::ConstPtrAttr>(
          [&tc](cir::ConstPtrAttr cpa) { return tc.isLegal(cpa.getType()); })
      .Case<cir::CXXCtorAttr>(
          [&tc](cir::CXXCtorAttr ca) { return tc.isLegal(ca.getType()); })
      .Case<cir::CXXDtorAttr>(
          [&tc](cir::CXXDtorAttr da) { return tc.isLegal(da.getType()); })
      .Case<cir::CXXAssignAttr>(
          [&tc](cir::CXXAssignAttr aa) { return tc.isLegal(aa.getType()); })

      // Collection attributes are legal if ALL of the attributes in them are
      // also legal.
      .Case<mlir::ArrayAttr>([&tc](mlir::ArrayAttr array) {
        return llvm::all_of(array.getValue(), [&tc](mlir::Attribute attr) {
          return isCXXABIAttributeLegal(tc, attr);
        });
      })
      .Case<mlir::DictionaryAttr>([&tc](mlir::DictionaryAttr dict) {
        return llvm::all_of(dict.getValue(), [&tc](mlir::NamedAttribute na) {
          return isCXXABIAttributeLegal(tc, na.getValue());
        });
      })
      // These attributes have sub-attributes that we should check for legality.
      .Case<cir::ConstArrayAttr>([&tc](cir::ConstArrayAttr array) {
        return tc.isLegal(array.getType()) &&
               isCXXABIAttributeLegal(tc, array.getElts());
      })
      .Case<cir::GlobalViewAttr>([&tc](cir::GlobalViewAttr gva) {
        return tc.isLegal(gva.getType()) &&
               isCXXABIAttributeLegal(tc, gva.getIndices());
      })
      .Case<cir::VTableAttr>([&tc](cir::VTableAttr vta) {
        return tc.isLegal(vta.getType()) &&
               isCXXABIAttributeLegal(tc, vta.getData());
      })
      .Case<cir::TypeInfoAttr>([&tc](cir::TypeInfoAttr tia) {
        return tc.isLegal(tia.getType()) &&
               isCXXABIAttributeLegal(tc, tia.getData());
      })
      .Case<cir::DynamicCastInfoAttr>([&tc](cir::DynamicCastInfoAttr dcia) {
        return isCXXABIAttributeLegal(tc, dcia.getSrcRtti()) &&
               isCXXABIAttributeLegal(tc, dcia.getDestRtti()) &&
               isCXXABIAttributeLegal(tc, dcia.getRuntimeFunc()) &&
               isCXXABIAttributeLegal(tc, dcia.getBadCastFunc());
      })
      .Case<cir::ConstRecordAttr>([&tc](cir::ConstRecordAttr cra) {
        return tc.isLegal(cra.getType()) &&
               isCXXABIAttributeLegal(tc, cra.getMembers());
      })
      // We did an audit of all of our attributes (both in OpenACC and CIR), so
      // it shouldn't be dangerous to consider everything we haven't considered
      // 'illegal'. Any 'new' attributes will end up asserting in
      // 'rewriteAttribute' to make sure we consider them here. Otherwise, we
      // wouldn't discover a problematic new attribute until it contains a
      // member/method.
      .Default(false);
}

mlir::Attribute rewriteAttribute(const mlir::TypeConverter &tc,
                                 mlir::MLIRContext *ctx, mlir::Attribute attr) {
  // If the attribute is legal, there is no reason to rewrite it. This also
  // filters out 'null' attributes.
  if (isCXXABIAttributeLegal(tc, attr))
    return attr;

  // This switch needs to be kept in sync with the potentially-legal type switch
  // from isCXXABIAttributeLegal. IF we miss any, this will end up causing
  // verification/transformation issues later, often in the form of
  // unrealized-conversion-casts.

  return llvm::TypeSwitch<mlir::Attribute, mlir::Attribute>(attr)
      // These attributes just have a type, so convert just the type.
      .Case<cir::ZeroAttr>([&tc](cir::ZeroAttr za) {
        return cir::ZeroAttr::get(tc.convertType(za.getType()));
      })
      .Case<cir::PoisonAttr>([&tc](cir::PoisonAttr pa) {
        return cir::PoisonAttr::get(tc.convertType(pa.getType()));
      })
      .Case<cir::UndefAttr>([&tc](cir::UndefAttr uda) {
        return cir::UndefAttr::get(tc.convertType(uda.getType()));
      })
      .Case<mlir::TypeAttr>([&tc](mlir::TypeAttr ta) {
        return mlir::TypeAttr::get(tc.convertType(ta.getValue()));
      })
      .Case<cir::ConstPtrAttr>([&tc](cir::ConstPtrAttr cpa) {
        return cir::ConstPtrAttr::get(tc.convertType(cpa.getType()),
                                      cpa.getValue());
      })
      .Case<cir::CXXCtorAttr>([&tc](cir::CXXCtorAttr ca) {
        return cir::CXXCtorAttr::get(tc.convertType(ca.getType()),
                                     ca.getCtorKind(), ca.getIsTrivial());
      })
      .Case<cir::CXXDtorAttr>([&tc](cir::CXXDtorAttr da) {
        return cir::CXXDtorAttr::get(tc.convertType(da.getType()),
                                     da.getIsTrivial());
      })
      .Case<cir::CXXAssignAttr>([&tc](cir::CXXAssignAttr aa) {
        return cir::CXXAssignAttr::get(tc.convertType(aa.getType()),
                                       aa.getAssignKind(), aa.getIsTrivial());
      })
      // Collection attributes need to transform all of the attributes inside of
      // them.
      .Case<mlir::ArrayAttr>([&tc, ctx](mlir::ArrayAttr array) {
        llvm::SmallVector<mlir::Attribute> elts;
        for (mlir::Attribute a : array.getValue())
          elts.push_back(rewriteAttribute(tc, ctx, a));
        return mlir::ArrayAttr::get(ctx, elts);
      })
      .Case<mlir::DictionaryAttr>([&tc, ctx](mlir::DictionaryAttr dict) {
        llvm::SmallVector<mlir::NamedAttribute> elts;
        for (mlir::NamedAttribute na : dict.getValue())
          elts.emplace_back(na.getName(),
                            rewriteAttribute(tc, ctx, na.getValue()));

        return mlir::DictionaryAttr::get(ctx, elts);
      })
      // These attributes have sub-attributes that need converting too.
      .Case<cir::ConstArrayAttr>([&tc, ctx](cir::ConstArrayAttr array) {
        return cir::ConstArrayAttr::get(
            ctx, tc.convertType(array.getType()),
            rewriteAttribute(tc, ctx, array.getElts()),
            array.getTrailingZerosNum());
      })
      .Case<cir::GlobalViewAttr>([&tc, ctx](cir::GlobalViewAttr gva) {
        return cir::GlobalViewAttr::get(
            tc.convertType(gva.getType()), gva.getSymbol(),
            mlir::cast<mlir::ArrayAttr>(
                rewriteAttribute(tc, ctx, gva.getIndices())));
      })
      .Case<cir::VTableAttr>([&tc, ctx](cir::VTableAttr vta) {
        return cir::VTableAttr::get(
            tc.convertType(vta.getType()),
            mlir::cast<mlir::ArrayAttr>(
                rewriteAttribute(tc, ctx, vta.getData())));
      })
      .Case<cir::TypeInfoAttr>([&tc, ctx](cir::TypeInfoAttr tia) {
        return cir::TypeInfoAttr::get(
            tc.convertType(tia.getType()),
            mlir::cast<mlir::ArrayAttr>(
                rewriteAttribute(tc, ctx, tia.getData())));
      })
      .Case<cir::DynamicCastInfoAttr>([&tc,
                                       ctx](cir::DynamicCastInfoAttr dcia) {
        return cir::DynamicCastInfoAttr::get(
            mlir::cast<cir::GlobalViewAttr>(
                rewriteAttribute(tc, ctx, dcia.getSrcRtti())),
            mlir::cast<cir::GlobalViewAttr>(
                rewriteAttribute(tc, ctx, dcia.getDestRtti())),
            dcia.getRuntimeFunc(), dcia.getBadCastFunc(), dcia.getOffsetHint());
      })
      .Case<cir::ConstRecordAttr>([&tc, ctx](cir::ConstRecordAttr cra) {
        return cir::ConstRecordAttr::get(
            ctx, tc.convertType(cra.getType()),
            mlir::cast<mlir::ArrayAttr>(
                rewriteAttribute(tc, ctx, cra.getMembers())));
      })
      .DefaultUnreachable("unrewritten illegal attribute kind");
}

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
    if (llvm::isa<cir::AllocaOp, cir::BaseDataMemberOp, cir::BaseMethodOp,
                  cir::CastOp, cir::CmpOp, cir::ConstantOp, cir::DeleteArrayOp,
                  cir::DerivedDataMemberOp, cir::DerivedMethodOp, cir::FuncOp,
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
    bool attrsLegal =
        llvm::all_of(op->getAttrs(), [typeConverter](mlir::NamedAttribute na) {
          return isCXXABIAttributeLegal(*typeConverter, na.getValue());
        });

    if (operandsAndResultsLegal && regionsLegal && attrsLegal) {
      // The operation does not have any CXXABI-dependent operands or results,
      // the match fails.
      return mlir::failure();
    }

    mlir::OperationState loweredOpState(op->getLoc(), op->getName());
    loweredOpState.addOperands(operands);
    loweredOpState.addSuccessors(op->getSuccessors());

    // Lower all attributes.
    llvm::SmallVector<mlir::NamedAttribute> attrs;
    for (const mlir::NamedAttribute &na : op->getAttrs())
      attrs.push_back(
          {na.getName(),
           rewriteAttribute(*typeConverter, op->getContext(), na.getValue())});
    loweredOpState.addAttributes(attrs);

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

  if (mlir::isa<cir::DataMemberType, cir::MethodType>(srcTy)) {
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
      if (mlir::isa<cir::DataMemberType>(srcTy))
        loweredResult = lowerModule->getCXXABI().lowerDataMemberToBoolCast(
            op, adaptor.getSrc(), rewriter);
      else
        loweredResult = lowerModule->getCXXABI().lowerMethodToBoolCast(
            op, adaptor.getSrc(), rewriter);
      rewriter.replaceOp(op, loweredResult);
      return mlir::success();
    }
    default:
      break;
    }
  }

  mlir::Value loweredResult = cir::CastOp::create(
      rewriter, op.getLoc(), getTypeConverter()->convertType(op.getType()),
      adaptor.getKind(), adaptor.getSrc());
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

// Helper function to lower a value for things like an initializer.
static mlir::TypedAttr lowerInitialValue(const LowerModule *lowerModule,
                                         const mlir::DataLayout &layout,
                                         const mlir::TypeConverter &tc,
                                         mlir::Type ty,
                                         mlir::Attribute initVal) {
  if (mlir::isa<cir::DataMemberType>(ty)) {
    auto dataMemberVal = mlir::cast_if_present<cir::DataMemberAttr>(initVal);
    return lowerModule->getCXXABI().lowerDataMemberConstant(dataMemberVal,
                                                            layout, tc);
  }
  if (mlir::isa<cir::MethodType>(ty)) {
    auto methodVal = mlir::cast_if_present<cir::MethodAttr>(initVal);
    return lowerModule->getCXXABI().lowerMethodConstant(methodVal, layout, tc);
  }

  if (auto arrTy = mlir::dyn_cast<cir::ArrayType>(ty)) {
    auto loweredArrTy = mlir::cast<cir::ArrayType>(tc.convertType(arrTy));

    if (!initVal)
      return {};

    if (auto zeroVal = mlir::dyn_cast_if_present<cir::ZeroAttr>(initVal))
      return cir::ZeroAttr::get(loweredArrTy);

    auto arrayVal = mlir::cast<cir::ConstArrayAttr>(initVal);

    // String-literal arrays store their bytes as a StringAttr in `elts`. The
    // backing i8 element type is never rewritten by the CXX ABI type
    // converter, so the attribute is already legal and can be passed through
    // unchanged.
    if (mlir::isa<mlir::StringAttr>(arrayVal.getElts())) {
      assert(loweredArrTy == arrTy &&
             "string-literal array type should not change under CXX ABI");
      return arrayVal;
    }

    auto arrayElts = mlir::cast<ArrayAttr>(arrayVal.getElts());
    SmallVector<mlir::Attribute> loweredElements;
    loweredElements.reserve(arrTy.getSize());
    for (const mlir::Attribute &attr : arrayElts) {
      auto typedAttr = cast<mlir::TypedAttr>(attr);
      loweredElements.push_back(lowerInitialValue(
          lowerModule, layout, tc, typedAttr.getType(), typedAttr));
    }

    return cir::ConstArrayAttr::get(
        loweredArrTy, mlir::ArrayAttr::get(ty.getContext(), loweredElements),
        arrayVal.getTrailingZerosNum());
  }

  if (auto recordTy = mlir::dyn_cast<cir::RecordType>(ty)) {
    auto convertedTy = mlir::cast<cir::RecordType>(tc.convertType(recordTy));

    if (auto recVal = mlir::dyn_cast_if_present<cir::ZeroAttr>(initVal))
      return cir::ZeroAttr::get(convertedTy);

    if (auto undefVal = mlir::dyn_cast_if_present<cir::UndefAttr>(initVal))
      return cir::UndefAttr::get(convertedTy);

    // This might not be possible from Clang directly, but we can get here with
    // hand-written IR.
    if (auto poisonVal = mlir::dyn_cast_if_present<cir::PoisonAttr>(initVal))
      return cir::PoisonAttr::get(convertedTy);

    if (auto recVal =
            mlir::dyn_cast_if_present<cir::ConstRecordAttr>(initVal)) {
      auto recordMembers = mlir::cast<ArrayAttr>(recVal.getMembers());

      SmallVector<mlir::Attribute> loweredMembers;
      loweredMembers.reserve(recordMembers.size());

      for (const mlir::Attribute &attr : recordMembers) {
        auto typedAttr = cast<mlir::TypedAttr>(attr);
        loweredMembers.push_back(lowerInitialValue(
            lowerModule, layout, tc, typedAttr.getType(), typedAttr));
      }

      return cir::ConstRecordAttr::get(
          convertedTy, mlir::ArrayAttr::get(ty.getContext(), loweredMembers));
    }

    assert(!initVal && "Record init val type not handled");
    return {};
  }

  // Pointers can contain record types, which can change.
  if (auto ptrTy = mlir::dyn_cast<cir::PointerType>(ty)) {
    auto convertedTy = mlir::cast<cir::PointerType>(tc.convertType(ptrTy));
    // pointers don't change other than their types.

    if (auto gva = mlir::dyn_cast_if_present<cir::GlobalViewAttr>(initVal))
      return cir::GlobalViewAttr::get(convertedTy, gva.getSymbol(),
                                      gva.getIndices());

    auto constPtr = mlir::cast_if_present<cir::ConstPtrAttr>(initVal);
    if (!constPtr)
      return {};
    return cir::ConstPtrAttr::get(convertedTy, constPtr.getValue());
  }

  assert(ty == tc.convertType(ty) &&
         "cir.global or constant operand is not an CXXABI-dependent type");

  // Every other type can be left alone.
  return cast<mlir::TypedAttr>(initVal);
}

mlir::LogicalResult CIRConstantOpABILowering::matchAndRewrite(
    cir::ConstantOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
  mlir::TypedAttr newValue = lowerInitialValue(
      lowerModule, layout, *getTypeConverter(), op.getType(), op.getValue());
  rewriter.replaceOpWithNewOp<ConstantOp>(op, newValue);
  return mlir::success();
}

mlir::LogicalResult CIRCmpOpABILowering::matchAndRewrite(
    cir::CmpOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type type = op.getLhs().getType();

  mlir::Value loweredResult;
  if (mlir::isa<cir::DataMemberType>(type))
    loweredResult = lowerModule->getCXXABI().lowerDataMemberCmp(
        op, adaptor.getLhs(), adaptor.getRhs(), rewriter);
  else if (mlir::isa<cir::MethodType>(type))
    loweredResult = lowerModule->getCXXABI().lowerMethodCmp(
        op, adaptor.getLhs(), adaptor.getRhs(), rewriter);
  else
    loweredResult = cir::CmpOp::create(
        rewriter, op.getLoc(), getTypeConverter()->convertType(op.getType()),
        adaptor.getKind(), adaptor.getLhs(), adaptor.getRhs());

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

  llvm::SmallVector<mlir::NamedAttribute> attrs;
  for (const mlir::NamedAttribute &na : op->getAttrs())
    attrs.push_back(
        {na.getName(), rewriteAttribute(*getTypeConverter(), op->getContext(),
                                        na.getValue())});

  loweredFuncOp->setAttrs(attrs);

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

  mlir::Attribute loweredInit = lowerInitialValue(
      lowerModule, layout, *getTypeConverter(), ty, op.getInitialValueAttr());

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

mlir::LogicalResult CIRBaseMethodOpABILowering::matchAndRewrite(
    cir::BaseMethodOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult =
      lowerModule->getCXXABI().lowerBaseMethod(op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

mlir::LogicalResult CIRDeleteArrayOpABILowering::matchAndRewrite(
    cir::DeleteArrayOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::FlatSymbolRefAttr deleteFn = op.getDeleteFnAttr();
  mlir::Location loc = op->getLoc();
  mlir::Value loweredAddress = adaptor.getAddress();

  cir::UsualDeleteParamsAttr deleteParams = op.getDeleteParams();
  bool cookieRequired = deleteParams.getSize();
  assert((deleteParams.getSize() || !op.getElementDtorAttr()) &&
         "Expected size parameter when dtor fn is provided!");

  if (deleteParams.getTypeAwareDelete() || deleteParams.getDestroyingDelete() ||
      deleteParams.getAlignment())
    return rewriter.notifyMatchFailure(
        op, "type-aware, destroying, or aligned delete not yet supported");

  const CIRCXXABI &cxxABI = lowerModule->getCXXABI();
  CIRBaseBuilderTy cirBuilder(rewriter);
  mlir::Value deletePtr;
  llvm::SmallVector<mlir::Value> callArgs;

  if (cookieRequired) {
    mlir::Value numElements;
    clang::CharUnits cookieSize;
    auto ptrTy = mlir::cast<cir::PointerType>(loweredAddress.getType());
    mlir::DataLayout dl(op->getParentOfType<mlir::ModuleOp>());

    cxxABI.readArrayCookie(loc, loweredAddress, dl, cirBuilder, numElements,
                           deletePtr, cookieSize);

    // If a dtor function is provided, create an array dtor operation.
    // This will get expanded during LoweringPrepare.
    mlir::FlatSymbolRefAttr dtorFn = op.getElementDtorAttr();
    if (dtorFn) {
      auto eltPtrTy = cir::PointerType::get(ptrTy.getPointee());
      cir::ArrayDtor::create(
          rewriter, loc, loweredAddress, numElements,
          [&](mlir::OpBuilder &b, mlir::Location l) {
            auto arg = b.getInsertionBlock()->addArgument(eltPtrTy, l);
            cir::CallOp::create(b, l, dtorFn, cir::VoidType(),
                                mlir::ValueRange{arg});
            cir::YieldOp::create(b, l);
          });
    }

    // Compute the total allocation size and add it to the call arguments.
    callArgs.push_back(deletePtr);
    uint64_t eltSizeBytes = dl.getTypeSizeInBits(ptrTy.getPointee()) / 8;
    unsigned ptrWidth =
        lowerModule->getTarget().getPointerWidth(clang::LangAS::Default);
    cir::IntType sizeTy = cirBuilder.getUIntNTy(ptrWidth);

    mlir::Value eltSizeVal = cir::ConstantOp::create(
        rewriter, loc, cir::IntAttr::get(sizeTy, eltSizeBytes));
    mlir::Value allocSize =
        cir::MulOp::create(rewriter, loc, sizeTy, eltSizeVal, numElements);
    mlir::Value cookieSizeVal = cir::ConstantOp::create(
        rewriter, loc, cir::IntAttr::get(sizeTy, cookieSize.getQuantity()));
    allocSize =
        cir::AddOp::create(rewriter, loc, sizeTy, allocSize, cookieSizeVal);
    callArgs.push_back(allocSize);
  } else {
    deletePtr = cir::CastOp::create(rewriter, loc, cirBuilder.getVoidPtrTy(),
                                    cir::CastKind::bitcast, loweredAddress);
    callArgs.push_back(deletePtr);
  }

  cir::CallOp::create(rewriter, loc, deleteFn, cir::VoidType(), callArgs);
  rewriter.eraseOp(op);
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

mlir::LogicalResult CIRDerivedMethodOpABILowering::matchAndRewrite(
    cir::DerivedMethodOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult = lowerModule->getCXXABI().lowerDerivedMethod(
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

mlir::LogicalResult CIRVTableGetTypeInfoOpABILowering::matchAndRewrite(
    cir::VTableGetTypeInfoOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult =
      lowerModule->getCXXABI().lowerVTableGetTypeInfo(op, rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

namespace {
// A small type to handle type conversion for the the CXXABILoweringPass.
// Even though this is a CIR-to-CIR pass, we are eliminating some CIR types.
// Most importantly, this pass solves recursive type conversion problems by
// keeping a call stack.
class CIRABITypeConverter : public mlir::TypeConverter {

  mlir::MLIRContext &context;

  // Recursive structure detection.
  // We store one entry per thread here, and rely on locking. This works the
  // same way as the LLVM-IR lowering does it, which has a similar problem.
  DenseMap<uint64_t, std::unique_ptr<SmallVector<cir::RecordType>>>
      conversionCallStack;
  llvm::sys::SmartRWMutex<true> callStackMutex;

  // In order to let us 'change the names' back after the fact, we collect them
  // along the way.  They should only be added/accessed via the thread-safe
  // functions below.
  llvm::SmallVector<cir::RecordType> convertedRecordTypes;
  llvm::sys::SmartRWMutex<true> recordTypeMutex;

  // This provides a stack for the RecordTypes being processed on the current
  // thread, which lets us solve recursive conversions. This implementation is
  // cribbed from the LLVMTypeConverter which solves a similar but not identical
  // problem.
  SmallVector<cir::RecordType> &getCurrentThreadRecursiveStack() {
    {
      // Most of the time, the entry already exists in the map.
      std::shared_lock<decltype(callStackMutex)> lock(callStackMutex,
                                                      std::defer_lock);
      if (context.isMultithreadingEnabled())
        lock.lock();
      auto recursiveStack = conversionCallStack.find(llvm::get_threadid());
      if (recursiveStack != conversionCallStack.end())
        return *recursiveStack->second;
    }

    // First time this thread gets here, we have to get an exclusive access to
    // insert in the map
    std::unique_lock<decltype(callStackMutex)> lock(callStackMutex);
    auto recursiveStackInserted = conversionCallStack.insert(
        std::make_pair(llvm::get_threadid(),
                       std::make_unique<SmallVector<cir::RecordType>>()));
    return *recursiveStackInserted.first->second;
  }

  void addConvertedRecordType(cir::RecordType rt) {
    std::unique_lock<decltype(recordTypeMutex)> lock(recordTypeMutex);
    convertedRecordTypes.push_back(rt);
  }

  llvm::SmallVector<mlir::Type> convertRecordMemberTypes(cir::RecordType type) {
    llvm::SmallVector<mlir::Type> loweredMemberTypes;
    loweredMemberTypes.reserve(type.getNumElements());

    if (mlir::failed(convertTypes(type.getMembers(), loweredMemberTypes)))
      return {};

    return loweredMemberTypes;
  }

  cir::RecordType convertRecordType(cir::RecordType type) {
    // Unnamed record types can't be referred to recursively, so we can just
    // convert this one. It also doesn't have uniqueness problems, so we can
    // just do a conversion on it.
    if (!type.getName())
      return cir::RecordType::get(
          type.getContext(), convertRecordMemberTypes(type), type.getPacked(),
          type.getPadded(), type.getKind());

    assert(!type.isIncomplete() || type.getMembers().empty());

    // If the type has already been converted, we can just return, since there
    // is nothing to do. Also, if it is incomplete, it can't have invalid
    // members! So we can skip transforming it.
    if (type.isIncomplete() || type.isABIConvertedRecord())
      return type;

    SmallVectorImpl<cir::RecordType> &recursiveStack =
        getCurrentThreadRecursiveStack();

    auto convertedType = cir::RecordType::get(
        type.getContext(), type.getABIConvertedName(), type.getKind());

    // This type has already been converted, just return it.
    if (convertedType.isComplete())
      return convertedType;

    // We put the existing 'type' into the vector if we're in the process of
    // converting it (and pop it when we're done).  To prevent recursion,
    // just return the 'incomplete' version, and the 'top level' version of this
    // call will call 'complete' on it.
    if (llvm::is_contained(recursiveStack, type))
      return convertedType;

    recursiveStack.push_back(type);
    llvm::scope_exit popConvertingType(
        [&recursiveStack]() { recursiveStack.pop_back(); });

    SmallVector<mlir::Type> convertedMembers = convertRecordMemberTypes(type);

    convertedType.complete(convertedMembers, type.getPacked(),
                           type.getPadded());
    addConvertedRecordType(convertedType);
    return convertedType;
  }

public:
  CIRABITypeConverter(mlir::MLIRContext &ctx, mlir::DataLayout &dataLayout,
                      cir::LowerModule &lowerModule)
      : context(ctx) {
    addConversion([&](mlir::Type type) -> mlir::Type { return type; });
    // This is necessary in order to convert CIR pointer types that are
    // pointing to CIR types that we are lowering in this pass.
    addConversion([&](cir::PointerType type) -> mlir::Type {
      mlir::Type loweredPointeeType = convertType(type.getPointee());
      if (!loweredPointeeType)
        return {};
      return cir::PointerType::get(type.getContext(), loweredPointeeType,
                                   type.getAddrSpace());
    });
    addConversion([&](cir::ArrayType type) -> mlir::Type {
      mlir::Type loweredElementType = convertType(type.getElementType());
      if (!loweredElementType)
        return {};
      return cir::ArrayType::get(loweredElementType, type.getSize());
    });

    addConversion([&](cir::DataMemberType type) -> mlir::Type {
      mlir::Type abiType =
          lowerModule.getCXXABI().lowerDataMemberType(type, *this);
      return convertType(abiType);
    });
    addConversion([&](cir::MethodType type) -> mlir::Type {
      mlir::Type abiType = lowerModule.getCXXABI().lowerMethodType(type, *this);
      return convertType(abiType);
    });
    // This is necessary in order to convert CIR function types that have
    // argument or return types that use CIR types that we are lowering in
    // this pass.
    addConversion([&](cir::FuncType type) -> mlir::Type {
      llvm::SmallVector<mlir::Type> loweredInputTypes;
      loweredInputTypes.reserve(type.getNumInputs());
      if (mlir::failed(convertTypes(type.getInputs(), loweredInputTypes)))
        return {};

      mlir::Type loweredReturnType = convertType(type.getReturnType());
      if (!loweredReturnType)
        return {};

      return cir::FuncType::get(loweredInputTypes, loweredReturnType,
                                /*isVarArg=*/type.getVarArg());
    });
    addConversion([&](cir::RecordType type) -> mlir::Type {
      return convertRecordType(type);
    });
  }

  void restoreRecordTypeNames() {
    std::unique_lock<decltype(recordTypeMutex)> lock(recordTypeMutex);

    for (auto rt : convertedRecordTypes)
      rt.removeABIConversionNamePrefix();
  }
};
} // namespace

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

        bool attrs = llvm::all_of(
            op->getAttrs(), [&typeConverter](const mlir::NamedAttribute &a) {
              return isCXXABIAttributeLegal(typeConverter, a.getValue());
            });

        return attrs &&
               std::all_of(op->getRegions().begin(), op->getRegions().end(),
                           [&typeConverter](mlir::Region &region) {
                             return typeConverter.isLegal(&region);
                           });
      });

  target.addDynamicallyLegalDialect<mlir::acc::OpenACCDialect>(
      [&typeConverter](mlir::Operation *op) {
        if (!typeConverter.isLegal(op))
          return false;

        bool attrs = llvm::all_of(
            op->getAttrs(), [&typeConverter](const mlir::NamedAttribute &a) {
              return isCXXABIAttributeLegal(typeConverter, a.getValue());
            });

        return attrs &&
               std::all_of(op->getRegions().begin(), op->getRegions().end(),
                           [&typeConverter](mlir::Region &region) {
                             return typeConverter.isLegal(&region);
                           });
      });

  // Some CIR ops needs special checking for legality
  target.addDynamicallyLegalOp<cir::FuncOp>([&typeConverter](cir::FuncOp op) {
    bool attrs = llvm::all_of(
        op->getAttrs(), [&typeConverter](const mlir::NamedAttribute &a) {
          return isCXXABIAttributeLegal(typeConverter, a.getValue());
        });

    return attrs && typeConverter.isLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<cir::GlobalOp>(
      [&typeConverter](cir::GlobalOp op) {
        return typeConverter.isLegal(op.getSymType());
      });
  // Operations that do not use any special types must be explicitly marked as
  // illegal to trigger processing here.
  target.addIllegalOp<cir::DeleteArrayOp>();
  target.addIllegalOp<cir::DynamicCastOp>();
  target.addIllegalOp<cir::VTableGetTypeInfoOp>();
}

//===----------------------------------------------------------------------===//
// The Pass
//===----------------------------------------------------------------------===//

// The applyPartialConversion function traverses blocks in the dominance order,
// so it does not lower and operations that are not reachachable from the
// operations passed in as arguments. Since we do need to lower such code in
// order to avoid verification errors occur, we cannot just pass the module op
// to applyPartialConversion. We must build a set of unreachable ops and
// explicitly add them, along with the module, to the vector we pass to
// applyPartialConversion.
//
// For instance, this CIR code:
//
//    cir.func @foo(%arg0: !s32i) -> !s32i {
//      %4 = cir.cast int_to_bool %arg0 : !s32i -> !cir.bool
//      cir.if %4 {
//        %5 = cir.const #cir.int<1> : !s32i
//        cir.return %5 : !s32i
//      } else {
//        %5 = cir.const #cir.int<0> : !s32i
//       cir.return %5 : !s32i
//      }
//      cir.return %arg0 : !s32i
//    }
//
// contains an unreachable return operation (the last one). After the CXXABI
// pass it will be placed into the unreachable block.  This will error because
// it will have not converted the types in the block, making the legalizer fail.
//
// In the future we may want to get rid of this function and use a DCE pass or
// something similar. But for now we need to guarantee the absence of the
// dialect verification errors. Note: We do the same in LowerToLLVM as well,
// this is a striaght copy/paste including most of the comment. We might wi sh
// to combine these if we don't want to do a DCE pass/etc.
static void collectUnreachable(mlir::Operation *parent,
                               llvm::SmallVector<mlir::Operation *> &ops) {

  llvm::SmallVector<mlir::Block *> unreachableBlocks;
  parent->walk([&](mlir::Block *blk) { // check
    if (blk->hasNoPredecessors() && !blk->isEntryBlock())
      unreachableBlocks.push_back(blk);
  });

  std::set<mlir::Block *> visited;
  for (mlir::Block *root : unreachableBlocks) {
    // We create a work list for each unreachable block.
    // Thus we traverse operations in some order.
    std::deque<mlir::Block *> workList;
    workList.push_back(root);

    while (!workList.empty()) {
      mlir::Block *blk = workList.back();
      workList.pop_back();
      if (visited.count(blk))
        continue;
      visited.emplace(blk);

      for (mlir::Operation &op : *blk)
        ops.push_back(&op);

      for (mlir::Block *succ : blk->getSuccessors())
        workList.push_back(succ);
    }
  }
}

void CXXABILoweringPass::runOnOperation() {
  auto mod = mlir::cast<mlir::ModuleOp>(getOperation());
  mlir::MLIRContext *ctx = mod.getContext();

  std::unique_ptr<cir::LowerModule> lowerModule = cir::createLowerModule(mod);
  // If lower module is not available, skip the ABI lowering pass.
  if (!lowerModule) {
    mod.emitWarning("Cannot create a CIR lower module, skipping the ")
        << getName() << " pass";
    return;
  }

  mlir::DataLayout dataLayout(mod);
  CIRABITypeConverter typeConverter(*ctx, dataLayout, *lowerModule);

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

  llvm::SmallVector<mlir::Operation *> ops;
  ops.push_back(mod);
  collectUnreachable(mod, ops);

  if (failed(mlir::applyPartialConversion(ops, target, std::move(patterns))))
    signalPassFailure();

  typeConverter.restoreRecordTypeNames();
}

std::unique_ptr<Pass> mlir::createCXXABILoweringPass() {
  return std::make_unique<CXXABILoweringPass>();
}
