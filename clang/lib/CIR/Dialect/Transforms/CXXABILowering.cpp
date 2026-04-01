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

#include "aiir/IR/PatternMatch.h"
#include "aiir/Interfaces/DataLayoutInterfaces.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/MissingFeatures.h"

using namespace aiir;
using namespace cir;

namespace aiir {
#define GEN_PASS_DEF_CXXABILOWERING
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace aiir

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
class CIRGenericCXXABILoweringPattern : public aiir::ConversionPattern {
public:
  CIRGenericCXXABILoweringPattern(aiir::AIIRContext *context,
                                  const aiir::TypeConverter &typeConverter)
      : aiir::ConversionPattern(typeConverter, MatchAnyOpTypeTag(),
                                /*benefit=*/1, context) {}

  aiir::LogicalResult
  matchAndRewrite(aiir::Operation *op, llvm::ArrayRef<aiir::Value> operands,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    // Do not match on operations that have dedicated ABI lowering rewrite rules
    if (llvm::isa<cir::AllocaOp, cir::BaseDataMemberOp, cir::BaseMethodOp,
                  cir::CastOp, cir::CmpOp, cir::ConstantOp, cir::DeleteArrayOp,
                  cir::DerivedDataMemberOp, cir::DerivedMethodOp, cir::FuncOp,
                  cir::GetMethodOp, cir::GetRuntimeMemberOp, cir::GlobalOp>(op))
      return aiir::failure();

    const aiir::TypeConverter *typeConverter = getTypeConverter();
    assert(typeConverter &&
           "CIRGenericCXXABILoweringPattern requires a type converter");
    bool operandsAndResultsLegal = typeConverter->isLegal(op);
    bool regionsLegal =
        std::all_of(op->getRegions().begin(), op->getRegions().end(),
                    [typeConverter](aiir::Region &region) {
                      return typeConverter->isLegal(&region);
                    });
    if (operandsAndResultsLegal && regionsLegal) {
      // The operation does not have any CXXABI-dependent operands or results,
      // the match fails.
      return aiir::failure();
    }

    assert(op->getNumRegions() == 0 && "CIRGenericCXXABILoweringPattern cannot "
                                       "deal with operations with regions");

    aiir::OperationState loweredOpState(op->getLoc(), op->getName());
    loweredOpState.addOperands(operands);
    loweredOpState.addAttributes(op->getAttrs());
    loweredOpState.addSuccessors(op->getSuccessors());

    // Lower all result types
    llvm::SmallVector<aiir::Type> loweredResultTypes;
    loweredResultTypes.reserve(op->getNumResults());
    for (aiir::Type result : op->getResultTypes())
      loweredResultTypes.push_back(typeConverter->convertType(result));
    loweredOpState.addTypes(loweredResultTypes);

    // Lower all regions
    for (aiir::Region &region : op->getRegions()) {
      aiir::Region *loweredRegion = loweredOpState.addRegion();
      rewriter.inlineRegionBefore(region, *loweredRegion, loweredRegion->end());
      if (aiir::failed(
              rewriter.convertRegionTypes(loweredRegion, *getTypeConverter())))
        return aiir::failure();
    }

    // Clone the operation with lowered operand types and result types
    aiir::Operation *loweredOp = rewriter.create(loweredOpState);

    rewriter.replaceOp(op, loweredOp);
    return aiir::success();
  }
};

} // namespace

aiir::LogicalResult CIRAllocaOpABILowering::matchAndRewrite(
    cir::AllocaOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type allocaPtrTy = op.getType();
  aiir::Type allocaTy = op.getAllocaType();
  aiir::Type loweredAllocaPtrTy = getTypeConverter()->convertType(allocaPtrTy);
  aiir::Type loweredAllocaTy = getTypeConverter()->convertType(allocaTy);

  cir::AllocaOp loweredOp = cir::AllocaOp::create(
      rewriter, op.getLoc(), loweredAllocaPtrTy, loweredAllocaTy, op.getName(),
      op.getAlignmentAttr(), /*dynAllocSize=*/adaptor.getDynAllocSize());
  loweredOp.setInit(op.getInit());
  loweredOp.setConstant(op.getConstant());
  loweredOp.setAnnotationsAttr(op.getAnnotationsAttr());

  rewriter.replaceOp(op, loweredOp);
  return aiir::success();
}

aiir::LogicalResult CIRCastOpABILowering::matchAndRewrite(
    cir::CastOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type srcTy = op.getSrc().getType();
  assert((aiir::isa<cir::DataMemberType, cir::MethodType>(srcTy)) &&
         "input to bitcast in ABI lowering must be a data member or method");

  switch (op.getKind()) {
  case cir::CastKind::bitcast: {
    aiir::Type destTy = getTypeConverter()->convertType(op.getType());
    aiir::Value loweredResult;
    if (aiir::isa<cir::DataMemberType>(srcTy))
      loweredResult = lowerModule->getCXXABI().lowerDataMemberBitcast(
          op, destTy, adaptor.getSrc(), rewriter);
    else
      loweredResult = lowerModule->getCXXABI().lowerMethodBitcast(
          op, destTy, adaptor.getSrc(), rewriter);
    rewriter.replaceOp(op, loweredResult);
    return aiir::success();
  }
  case cir::CastKind::member_ptr_to_bool: {
    aiir::Value loweredResult;
    if (aiir::isa<cir::MethodType>(srcTy))
      loweredResult = lowerModule->getCXXABI().lowerMethodToBoolCast(
          op, adaptor.getSrc(), rewriter);
    else
      loweredResult = lowerModule->getCXXABI().lowerDataMemberToBoolCast(
          op, adaptor.getSrc(), rewriter);
    rewriter.replaceOp(op, loweredResult);
    return aiir::success();
  }
  default:
    break;
  }

  return aiir::failure();
}
// Helper function to lower a value for things like an initializer.
static aiir::TypedAttr lowerInitialValue(const LowerModule *lowerModule,
                                         const aiir::DataLayout &layout,
                                         const aiir::TypeConverter &tc,
                                         aiir::Type ty,
                                         aiir::Attribute initVal) {
  if (aiir::isa<cir::DataMemberType>(ty)) {
    auto dataMemberVal = aiir::cast_if_present<cir::DataMemberAttr>(initVal);
    return lowerModule->getCXXABI().lowerDataMemberConstant(dataMemberVal,
                                                            layout, tc);
  }
  if (aiir::isa<cir::MethodType>(ty)) {
    auto methodVal = aiir::cast_if_present<cir::MethodAttr>(initVal);
    return lowerModule->getCXXABI().lowerMethodConstant(methodVal, layout, tc);
  }

  if (auto arrTy = aiir::dyn_cast<cir::ArrayType>(ty)) {
    auto loweredArrTy = aiir::cast<cir::ArrayType>(tc.convertType(arrTy));
    // TODO(cir): there are other types that can appear here inside of record
    // members that we should handle. Those will come in a follow-up patch to
    // minimize changes here.
    if (!initVal)
      return {};
    auto arrayVal = aiir::cast<cir::ConstArrayAttr>(initVal);
    auto arrayElts = aiir::cast<ArrayAttr>(arrayVal.getElts());
    SmallVector<aiir::Attribute> loweredElements;
    loweredElements.reserve(arrTy.getSize());
    for (const aiir::Attribute &attr : arrayElts) {
      auto typedAttr = cast<aiir::TypedAttr>(attr);
      loweredElements.push_back(lowerInitialValue(
          lowerModule, layout, tc, typedAttr.getType(), typedAttr));
    }

    return cir::ConstArrayAttr::get(
        loweredArrTy, aiir::ArrayAttr::get(ty.getContext(), loweredElements),
        arrayVal.getTrailingZerosNum());
  }

  llvm_unreachable("inputs to cir.global/constant in ABI lowering must be data "
                   "member or method");
}

aiir::LogicalResult CIRConstantOpABILowering::matchAndRewrite(
    cir::ConstantOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {

  aiir::DataLayout layout(op->getParentOfType<aiir::ModuleOp>());
  aiir::TypedAttr newValue = lowerInitialValue(
      lowerModule, layout, *getTypeConverter(), op.getType(), op.getValue());
  rewriter.replaceOpWithNewOp<ConstantOp>(op, newValue);
  return aiir::success();
}

aiir::LogicalResult CIRCmpOpABILowering::matchAndRewrite(
    cir::CmpOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type type = op.getLhs().getType();
  assert((aiir::isa<cir::DataMemberType, cir::MethodType>(type)) &&
         "input to cmp in ABI lowering must be a data member or method");

  aiir::Value loweredResult;
  if (aiir::isa<cir::DataMemberType>(type))
    loweredResult = lowerModule->getCXXABI().lowerDataMemberCmp(
        op, adaptor.getLhs(), adaptor.getRhs(), rewriter);
  else
    loweredResult = lowerModule->getCXXABI().lowerMethodCmp(
        op, adaptor.getLhs(), adaptor.getRhs(), rewriter);

  rewriter.replaceOp(op, loweredResult);
  return aiir::success();
}

aiir::LogicalResult CIRFuncOpABILowering::matchAndRewrite(
    cir::FuncOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  cir::FuncType opFuncType = op.getFunctionType();
  aiir::TypeConverter::SignatureConversion signatureConversion(
      opFuncType.getNumInputs());

  for (const auto &[i, argType] : llvm::enumerate(opFuncType.getInputs())) {
    aiir::Type loweredArgType = getTypeConverter()->convertType(argType);
    if (!loweredArgType)
      return aiir::failure();
    signatureConversion.addInputs(i, loweredArgType);
  }

  aiir::Type loweredResultType =
      getTypeConverter()->convertType(opFuncType.getReturnType());
  if (!loweredResultType)
    return aiir::failure();

  auto loweredFuncType =
      cir::FuncType::get(signatureConversion.getConvertedTypes(),
                         loweredResultType, /*isVarArg=*/opFuncType.isVarArg());

  // Create a new cir.func operation for the CXXABI-lowered function.
  cir::FuncOp loweredFuncOp = rewriter.cloneWithoutRegions(op);
  loweredFuncOp.setFunctionType(loweredFuncType);
  rewriter.inlineRegionBefore(op.getBody(), loweredFuncOp.getBody(),
                              loweredFuncOp.end());
  if (aiir::failed(rewriter.convertRegionTypes(
          &loweredFuncOp.getBody(), *getTypeConverter(), &signatureConversion)))
    return aiir::failure();

  rewriter.eraseOp(op);
  return aiir::success();
}

aiir::LogicalResult CIRGlobalOpABILowering::matchAndRewrite(
    cir::GlobalOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type ty = op.getSymType();
  aiir::Type loweredTy = getTypeConverter()->convertType(ty);
  if (!loweredTy)
    return aiir::failure();

  aiir::DataLayout layout(op->getParentOfType<aiir::ModuleOp>());

  aiir::Attribute loweredInit = lowerInitialValue(
      lowerModule, layout, *getTypeConverter(), ty, op.getInitialValueAttr());

  auto newOp = aiir::cast<cir::GlobalOp>(rewriter.clone(*op.getOperation()));
  newOp.setInitialValueAttr(loweredInit);
  newOp.setSymType(loweredTy);
  rewriter.replaceOp(op, newOp);
  return aiir::success();
}

aiir::LogicalResult CIRBaseDataMemberOpABILowering::matchAndRewrite(
    cir::BaseDataMemberOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value loweredResult = lowerModule->getCXXABI().lowerBaseDataMember(
      op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return aiir::success();
}

aiir::LogicalResult CIRBaseMethodOpABILowering::matchAndRewrite(
    cir::BaseMethodOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value loweredResult =
      lowerModule->getCXXABI().lowerBaseMethod(op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return aiir::success();
}

aiir::LogicalResult CIRDeleteArrayOpABILowering::matchAndRewrite(
    cir::DeleteArrayOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::FlatSymbolRefAttr deleteFn = op.getDeleteFnAttr();
  aiir::Location loc = op->getLoc();
  aiir::Value loweredAddress = adaptor.getAddress();

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
  aiir::Value deletePtr;
  llvm::SmallVector<aiir::Value> callArgs;

  if (cookieRequired) {
    aiir::Value numElements;
    clang::CharUnits cookieSize;
    auto ptrTy = aiir::cast<cir::PointerType>(loweredAddress.getType());
    aiir::DataLayout dl(op->getParentOfType<aiir::ModuleOp>());

    cxxABI.readArrayCookie(loc, loweredAddress, dl, cirBuilder, numElements,
                           deletePtr, cookieSize);

    // If a dtor function is provided, create an array dtor operation.
    // This will get expanded during LoweringPrepare.
    aiir::FlatSymbolRefAttr dtorFn = op.getElementDtorAttr();
    if (dtorFn) {
      auto eltPtrTy = cir::PointerType::get(ptrTy.getPointee());
      cir::ArrayDtor::create(
          rewriter, loc, loweredAddress, numElements,
          [&](aiir::OpBuilder &b, aiir::Location l) {
            auto arg = b.getInsertionBlock()->addArgument(eltPtrTy, l);
            cir::CallOp::create(b, l, dtorFn, cir::VoidType(),
                                aiir::ValueRange{arg});
            cir::YieldOp::create(b, l);
          });
    }

    // Compute the total allocation size and add it to the call arguments.
    callArgs.push_back(deletePtr);
    uint64_t eltSizeBytes = dl.getTypeSizeInBits(ptrTy.getPointee()) / 8;
    unsigned ptrWidth =
        lowerModule->getTarget().getPointerWidth(clang::LangAS::Default);
    cir::IntType sizeTy = cirBuilder.getUIntNTy(ptrWidth);

    aiir::Value eltSizeVal = cir::ConstantOp::create(
        rewriter, loc, cir::IntAttr::get(sizeTy, eltSizeBytes));
    aiir::Value allocSize =
        cir::MulOp::create(rewriter, loc, sizeTy, eltSizeVal, numElements);
    aiir::Value cookieSizeVal = cir::ConstantOp::create(
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
  return aiir::success();
}

aiir::LogicalResult CIRDerivedDataMemberOpABILowering::matchAndRewrite(
    cir::DerivedDataMemberOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value loweredResult = lowerModule->getCXXABI().lowerDerivedDataMember(
      op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return aiir::success();
}

aiir::LogicalResult CIRDerivedMethodOpABILowering::matchAndRewrite(
    cir::DerivedMethodOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value loweredResult = lowerModule->getCXXABI().lowerDerivedMethod(
      op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return aiir::success();
}

aiir::LogicalResult CIRDynamicCastOpABILowering::matchAndRewrite(
    cir::DynamicCastOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value loweredResult =
      lowerModule->getCXXABI().lowerDynamicCast(op, rewriter);
  rewriter.replaceOp(op, loweredResult);
  return aiir::success();
}

aiir::LogicalResult CIRGetMethodOpABILowering::matchAndRewrite(
    cir::GetMethodOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value callee;
  aiir::Value thisArg;
  lowerModule->getCXXABI().lowerGetMethod(
      op, callee, thisArg, adaptor.getMethod(), adaptor.getObject(), rewriter);
  rewriter.replaceOp(op, {callee, thisArg});
  return aiir::success();
}

aiir::LogicalResult CIRGetRuntimeMemberOpABILowering::matchAndRewrite(
    cir::GetRuntimeMemberOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Type resTy = getTypeConverter()->convertType(op.getType());
  aiir::Operation *newOp = lowerModule->getCXXABI().lowerGetRuntimeMember(
      op, resTy, adaptor.getAddr(), adaptor.getMember(), rewriter);
  rewriter.replaceOp(op, newOp);
  return aiir::success();
}

aiir::LogicalResult CIRVTableGetTypeInfoOpABILowering::matchAndRewrite(
    cir::VTableGetTypeInfoOp op, OpAdaptor adaptor,
    aiir::ConversionPatternRewriter &rewriter) const {
  aiir::Value loweredResult =
      lowerModule->getCXXABI().lowerVTableGetTypeInfo(op, rewriter);
  rewriter.replaceOp(op, loweredResult);
  return aiir::success();
}

// A type to handle type conversion for the CXXABILowering pass.
class CIRABITypeConverter : public aiir::TypeConverter {
public:
  CIRABITypeConverter(aiir::DataLayout &dataLayout,
                      cir::LowerModule &lowerModule) {
    addConversion([&](aiir::Type type) -> aiir::Type { return type; });
    // This is necessary in order to convert CIR pointer types that are
    // pointing to CIR types that we are lowering in this pass.
    addConversion([&](cir::PointerType type) -> aiir::Type {
      aiir::Type loweredPointeeType = convertType(type.getPointee());
      if (!loweredPointeeType)
        return {};
      return cir::PointerType::get(type.getContext(), loweredPointeeType,
                                   type.getAddrSpace());
    });
    addConversion([&](cir::ArrayType type) -> aiir::Type {
      aiir::Type loweredElementType = convertType(type.getElementType());
      if (!loweredElementType)
        return {};
      return cir::ArrayType::get(loweredElementType, type.getSize());
    });

    addConversion([&](cir::DataMemberType type) -> aiir::Type {
      aiir::Type abiType =
          lowerModule.getCXXABI().lowerDataMemberType(type, *this);
      return convertType(abiType);
    });
    addConversion([&](cir::MethodType type) -> aiir::Type {
      aiir::Type abiType = lowerModule.getCXXABI().lowerMethodType(type, *this);
      return convertType(abiType);
    });
    // This is necessary in order to convert CIR function types that have
    // argument or return types that use CIR types that we are lowering in
    // this pass.
    addConversion([&](cir::FuncType type) -> aiir::Type {
      llvm::SmallVector<aiir::Type> loweredInputTypes;
      loweredInputTypes.reserve(type.getNumInputs());
      if (aiir::failed(convertTypes(type.getInputs(), loweredInputTypes)))
        return {};

      aiir::Type loweredReturnType = convertType(type.getReturnType());
      if (!loweredReturnType)
        return {};

      return cir::FuncType::get(loweredInputTypes, loweredReturnType,
                                /*isVarArg=*/type.getVarArg());
    });
  }
};

static void
populateCXXABIConversionTarget(aiir::ConversionTarget &target,
                               const aiir::TypeConverter &typeConverter) {
  target.addLegalOp<aiir::ModuleOp>();

  // The ABI lowering pass is interested in CIR operations with operands or
  // results of CXXABI-dependent types, or CIR operations with regions whose
  // block arguments are of CXXABI-dependent types.
  target.addDynamicallyLegalDialect<cir::CIRDialect>(
      [&typeConverter](aiir::Operation *op) {
        if (!typeConverter.isLegal(op))
          return false;
        return std::all_of(op->getRegions().begin(), op->getRegions().end(),
                           [&typeConverter](aiir::Region &region) {
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
static void collectUnreachable(aiir::Operation *parent,
                               llvm::SmallVector<aiir::Operation *> &ops) {

  llvm::SmallVector<aiir::Block *> unreachableBlocks;
  parent->walk([&](aiir::Block *blk) { // check
    if (blk->hasNoPredecessors() && !blk->isEntryBlock())
      unreachableBlocks.push_back(blk);
  });

  std::set<aiir::Block *> visited;
  for (aiir::Block *root : unreachableBlocks) {
    // We create a work list for each unreachable block.
    // Thus we traverse operations in some order.
    std::deque<aiir::Block *> workList;
    workList.push_back(root);

    while (!workList.empty()) {
      aiir::Block *blk = workList.back();
      workList.pop_back();
      if (visited.count(blk))
        continue;
      visited.emplace(blk);

      for (aiir::Operation &op : *blk)
        ops.push_back(&op);

      for (aiir::Block *succ : blk->getSuccessors())
        workList.push_back(succ);
    }
  }
}

void CXXABILoweringPass::runOnOperation() {
  auto mod = aiir::cast<aiir::ModuleOp>(getOperation());
  aiir::AIIRContext *ctx = mod.getContext();

  std::unique_ptr<cir::LowerModule> lowerModule = cir::createLowerModule(mod);
  // If lower module is not available, skip the ABI lowering pass.
  if (!lowerModule) {
    mod.emitWarning("Cannot create a CIR lower module, skipping the ")
        << getName() << " pass";
    return;
  }

  aiir::DataLayout dataLayout(mod);
  CIRABITypeConverter typeConverter(dataLayout, *lowerModule);

  aiir::RewritePatternSet patterns(ctx);
  patterns.add<CIRGenericCXXABILoweringPattern>(patterns.getContext(),
                                                typeConverter);
  patterns.add<
#define GET_ABI_LOWERING_PATTERNS_LIST
#include "clang/CIR/Dialect/IR/CIRLowering.inc"
#undef GET_ABI_LOWERING_PATTERNS_LIST
      >(patterns.getContext(), typeConverter, dataLayout, *lowerModule);

  aiir::ConversionTarget target(*ctx);
  populateCXXABIConversionTarget(target, typeConverter);

  llvm::SmallVector<aiir::Operation *> ops;
  ops.push_back(mod);
  collectUnreachable(mod, ops);

  if (failed(aiir::applyPartialConversion(ops, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> aiir::createCXXABILoweringPass() {
  return std::make_unique<CXXABILoweringPass>();
}
