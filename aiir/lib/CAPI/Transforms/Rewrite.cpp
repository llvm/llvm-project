//===- Rewrite.cpp - C API for Rewrite Patterns ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Rewrite.h"

#include "aiir-c/Support.h"
#include "aiir-c/Transforms.h"
#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/Rewrite.h"
#include "aiir/CAPI/Support.h"
#include "aiir/CAPI/Wrap.h"
#include "aiir/IR/Attributes.h"
#include "aiir/IR/PDLPatternMatch.h.inc"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Rewrite/FrozenRewritePatternSet.h"
#include "aiir/Transforms/DialectConversion.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
#include "aiir/Transforms/WalkPatternRewriteDriver.h"

using namespace aiir;

//===----------------------------------------------------------------------===//
/// RewriterBase API inherited from OpBuilder
//===----------------------------------------------------------------------===//

AiirContext aiirRewriterBaseGetContext(AiirRewriterBase rewriter) {
  return wrap(unwrap(rewriter)->getContext());
}

//===----------------------------------------------------------------------===//
/// Insertion points methods
//===----------------------------------------------------------------------===//

void aiirRewriterBaseClearInsertionPoint(AiirRewriterBase rewriter) {
  unwrap(rewriter)->clearInsertionPoint();
}

void aiirRewriterBaseSetInsertionPointBefore(AiirRewriterBase rewriter,
                                             AiirOperation op) {
  unwrap(rewriter)->setInsertionPoint(unwrap(op));
}

void aiirRewriterBaseSetInsertionPointAfter(AiirRewriterBase rewriter,
                                            AiirOperation op) {
  unwrap(rewriter)->setInsertionPointAfter(unwrap(op));
}

void aiirRewriterBaseSetInsertionPointAfterValue(AiirRewriterBase rewriter,
                                                 AiirValue value) {
  unwrap(rewriter)->setInsertionPointAfterValue(unwrap(value));
}

void aiirRewriterBaseSetInsertionPointToStart(AiirRewriterBase rewriter,
                                              AiirBlock block) {
  unwrap(rewriter)->setInsertionPointToStart(unwrap(block));
}

void aiirRewriterBaseSetInsertionPointToEnd(AiirRewriterBase rewriter,
                                            AiirBlock block) {
  unwrap(rewriter)->setInsertionPointToEnd(unwrap(block));
}

AiirBlock aiirRewriterBaseGetInsertionBlock(AiirRewriterBase rewriter) {
  return wrap(unwrap(rewriter)->getInsertionBlock());
}

AiirBlock aiirRewriterBaseGetBlock(AiirRewriterBase rewriter) {
  return wrap(unwrap(rewriter)->getBlock());
}

AiirOperation
aiirRewriterBaseGetOperationAfterInsertion(AiirRewriterBase rewriter) {
  aiir::RewriterBase *base = unwrap(rewriter);
  aiir::Block *block = base->getInsertionBlock();
  aiir::Block::iterator it = base->getInsertionPoint();
  if (it == block->end())
    return {nullptr};

  return wrap(std::addressof(*it));
}

//===----------------------------------------------------------------------===//
/// Block and operation creation/insertion/cloning
//===----------------------------------------------------------------------===//

AiirBlock aiirRewriterBaseCreateBlockBefore(AiirRewriterBase rewriter,
                                            AiirBlock insertBefore,
                                            intptr_t nArgTypes,
                                            AiirType const *argTypes,
                                            AiirLocation const *locations) {
  SmallVector<Type, 4> args;
  ArrayRef<Type> unwrappedArgs = unwrapList(nArgTypes, argTypes, args);
  SmallVector<Location, 4> locs;
  ArrayRef<Location> unwrappedLocs = unwrapList(nArgTypes, locations, locs);
  return wrap(unwrap(rewriter)->createBlock(unwrap(insertBefore), unwrappedArgs,
                                            unwrappedLocs));
}

AiirOperation aiirRewriterBaseInsert(AiirRewriterBase rewriter,
                                     AiirOperation op) {
  return wrap(unwrap(rewriter)->insert(unwrap(op)));
}

// Other methods of OpBuilder

AiirOperation aiirRewriterBaseClone(AiirRewriterBase rewriter,
                                    AiirOperation op) {
  return wrap(unwrap(rewriter)->clone(*unwrap(op)));
}

AiirOperation aiirRewriterBaseCloneWithoutRegions(AiirRewriterBase rewriter,
                                                  AiirOperation op) {
  return wrap(unwrap(rewriter)->cloneWithoutRegions(*unwrap(op)));
}

void aiirRewriterBaseCloneRegionBefore(AiirRewriterBase rewriter,
                                       AiirRegion region, AiirBlock before) {

  unwrap(rewriter)->cloneRegionBefore(*unwrap(region), unwrap(before));
}

//===----------------------------------------------------------------------===//
/// RewriterBase API
//===----------------------------------------------------------------------===//

void aiirRewriterBaseInlineRegionBefore(AiirRewriterBase rewriter,
                                        AiirRegion region, AiirBlock before) {
  unwrap(rewriter)->inlineRegionBefore(*unwrap(region), unwrap(before));
}

void aiirRewriterBaseReplaceOpWithValues(AiirRewriterBase rewriter,
                                         AiirOperation op, intptr_t nValues,
                                         AiirValue const *values) {
  SmallVector<Value, 4> vals;
  ArrayRef<Value> unwrappedVals = unwrapList(nValues, values, vals);
  unwrap(rewriter)->replaceOp(unwrap(op), unwrappedVals);
}

void aiirRewriterBaseReplaceOpWithOperation(AiirRewriterBase rewriter,
                                            AiirOperation op,
                                            AiirOperation newOp) {
  unwrap(rewriter)->replaceOp(unwrap(op), unwrap(newOp));
}

void aiirRewriterBaseEraseOp(AiirRewriterBase rewriter, AiirOperation op) {
  unwrap(rewriter)->eraseOp(unwrap(op));
}

void aiirRewriterBaseEraseBlock(AiirRewriterBase rewriter, AiirBlock block) {
  unwrap(rewriter)->eraseBlock(unwrap(block));
}

void aiirRewriterBaseInlineBlockBefore(AiirRewriterBase rewriter,
                                       AiirBlock source, AiirOperation op,
                                       intptr_t nArgValues,
                                       AiirValue const *argValues) {
  SmallVector<Value, 4> vals;
  ArrayRef<Value> unwrappedVals = unwrapList(nArgValues, argValues, vals);

  unwrap(rewriter)->inlineBlockBefore(unwrap(source), unwrap(op),
                                      unwrappedVals);
}

void aiirRewriterBaseMergeBlocks(AiirRewriterBase rewriter, AiirBlock source,
                                 AiirBlock dest, intptr_t nArgValues,
                                 AiirValue const *argValues) {
  SmallVector<Value, 4> args;
  ArrayRef<Value> unwrappedArgs = unwrapList(nArgValues, argValues, args);
  unwrap(rewriter)->mergeBlocks(unwrap(source), unwrap(dest), unwrappedArgs);
}

void aiirRewriterBaseMoveOpBefore(AiirRewriterBase rewriter, AiirOperation op,
                                  AiirOperation existingOp) {
  unwrap(rewriter)->moveOpBefore(unwrap(op), unwrap(existingOp));
}

void aiirRewriterBaseMoveOpAfter(AiirRewriterBase rewriter, AiirOperation op,
                                 AiirOperation existingOp) {
  unwrap(rewriter)->moveOpAfter(unwrap(op), unwrap(existingOp));
}

void aiirRewriterBaseMoveBlockBefore(AiirRewriterBase rewriter, AiirBlock block,
                                     AiirBlock existingBlock) {
  unwrap(rewriter)->moveBlockBefore(unwrap(block), unwrap(existingBlock));
}

void aiirRewriterBaseStartOpModification(AiirRewriterBase rewriter,
                                         AiirOperation op) {
  unwrap(rewriter)->startOpModification(unwrap(op));
}

void aiirRewriterBaseFinalizeOpModification(AiirRewriterBase rewriter,
                                            AiirOperation op) {
  unwrap(rewriter)->finalizeOpModification(unwrap(op));
}

void aiirRewriterBaseCancelOpModification(AiirRewriterBase rewriter,
                                          AiirOperation op) {
  unwrap(rewriter)->cancelOpModification(unwrap(op));
}

void aiirRewriterBaseReplaceAllUsesWith(AiirRewriterBase rewriter,
                                        AiirValue from, AiirValue to) {
  unwrap(rewriter)->replaceAllUsesWith(unwrap(from), unwrap(to));
}

void aiirRewriterBaseReplaceAllValueRangeUsesWith(AiirRewriterBase rewriter,
                                                  intptr_t nValues,
                                                  AiirValue const *from,
                                                  AiirValue const *to) {
  SmallVector<Value, 4> fromVals;
  ArrayRef<Value> unwrappedFromVals = unwrapList(nValues, from, fromVals);
  SmallVector<Value, 4> toVals;
  ArrayRef<Value> unwrappedToVals = unwrapList(nValues, to, toVals);
  unwrap(rewriter)->replaceAllUsesWith(unwrappedFromVals, unwrappedToVals);
}

void aiirRewriterBaseReplaceAllOpUsesWithValueRange(AiirRewriterBase rewriter,
                                                    AiirOperation from,
                                                    intptr_t nTo,
                                                    AiirValue const *to) {
  SmallVector<Value, 4> toVals;
  ArrayRef<Value> unwrappedToVals = unwrapList(nTo, to, toVals);
  unwrap(rewriter)->replaceAllOpUsesWith(unwrap(from), unwrappedToVals);
}

void aiirRewriterBaseReplaceAllOpUsesWithOperation(AiirRewriterBase rewriter,
                                                   AiirOperation from,
                                                   AiirOperation to) {
  unwrap(rewriter)->replaceAllOpUsesWith(unwrap(from), unwrap(to));
}

void aiirRewriterBaseReplaceOpUsesWithinBlock(AiirRewriterBase rewriter,
                                              AiirOperation op,
                                              intptr_t nNewValues,
                                              AiirValue const *newValues,
                                              AiirBlock block) {
  SmallVector<Value, 4> vals;
  ArrayRef<Value> unwrappedVals = unwrapList(nNewValues, newValues, vals);
  unwrap(rewriter)->replaceOpUsesWithinBlock(unwrap(op), unwrappedVals,
                                             unwrap(block));
}

void aiirRewriterBaseReplaceAllUsesExcept(AiirRewriterBase rewriter,
                                          AiirValue from, AiirValue to,
                                          AiirOperation exceptedUser) {
  unwrap(rewriter)->replaceAllUsesExcept(unwrap(from), unwrap(to),
                                         unwrap(exceptedUser));
}

//===----------------------------------------------------------------------===//
/// IRRewriter API
//===----------------------------------------------------------------------===//

AiirRewriterBase aiirIRRewriterCreate(AiirContext context) {
  return wrap(new IRRewriter(unwrap(context)));
}

AiirRewriterBase aiirIRRewriterCreateFromOp(AiirOperation op) {
  return wrap(new IRRewriter(unwrap(op)));
}

void aiirIRRewriterDestroy(AiirRewriterBase rewriter) {
  delete static_cast<IRRewriter *>(unwrap(rewriter));
}

//===----------------------------------------------------------------------===//
/// RewritePatternSet and FrozenRewritePatternSet API
//===----------------------------------------------------------------------===//

AiirFrozenRewritePatternSet
aiirFreezeRewritePattern(AiirRewritePatternSet set) {
  auto *m = new aiir::FrozenRewritePatternSet(std::move(*unwrap(set)));
  set.ptr = nullptr;
  return wrap(m);
}

void aiirFrozenRewritePatternSetDestroy(AiirFrozenRewritePatternSet set) {
  delete unwrap(set);
  set.ptr = nullptr;
}

//===----------------------------------------------------------------------===//
/// GreedyRewriteDriverConfig API
//===----------------------------------------------------------------------===//

inline aiir::GreedyRewriteConfig *unwrap(AiirGreedyRewriteDriverConfig config) {
  assert(config.ptr && "unexpected null config");
  return static_cast<aiir::GreedyRewriteConfig *>(config.ptr);
}

inline AiirGreedyRewriteDriverConfig wrap(aiir::GreedyRewriteConfig *config) {
  return {config};
}

AiirGreedyRewriteDriverConfig aiirGreedyRewriteDriverConfigCreate() {
  return wrap(new aiir::GreedyRewriteConfig());
}

void aiirGreedyRewriteDriverConfigDestroy(
    AiirGreedyRewriteDriverConfig config) {
  delete unwrap(config);
}

void aiirGreedyRewriteDriverConfigSetMaxIterations(
    AiirGreedyRewriteDriverConfig config, int64_t maxIterations) {
  unwrap(config)->setMaxIterations(maxIterations);
}

void aiirGreedyRewriteDriverConfigSetMaxNumRewrites(
    AiirGreedyRewriteDriverConfig config, int64_t maxNumRewrites) {
  unwrap(config)->setMaxNumRewrites(maxNumRewrites);
}

void aiirGreedyRewriteDriverConfigSetUseTopDownTraversal(
    AiirGreedyRewriteDriverConfig config, bool useTopDownTraversal) {
  unwrap(config)->setUseTopDownTraversal(useTopDownTraversal);
}

void aiirGreedyRewriteDriverConfigEnableFolding(
    AiirGreedyRewriteDriverConfig config, bool enable) {
  unwrap(config)->enableFolding(enable);
}

void aiirGreedyRewriteDriverConfigSetStrictness(
    AiirGreedyRewriteDriverConfig config,
    AiirGreedyRewriteStrictness strictness) {
  aiir::GreedyRewriteStrictness cppStrictness;
  switch (strictness) {
  case AIIR_GREEDY_REWRITE_STRICTNESS_ANY_OP:
    cppStrictness = aiir::GreedyRewriteStrictness::AnyOp;
    break;
  case AIIR_GREEDY_REWRITE_STRICTNESS_EXISTING_AND_NEW_OPS:
    cppStrictness = aiir::GreedyRewriteStrictness::ExistingAndNewOps;
    break;
  case AIIR_GREEDY_REWRITE_STRICTNESS_EXISTING_OPS:
    cppStrictness = aiir::GreedyRewriteStrictness::ExistingOps;
    break;
  }
  unwrap(config)->setStrictness(cppStrictness);
}

void aiirGreedyRewriteDriverConfigSetRegionSimplificationLevel(
    AiirGreedyRewriteDriverConfig config, AiirGreedySimplifyRegionLevel level) {
  aiir::GreedySimplifyRegionLevel cppLevel;
  switch (level) {
  case AIIR_GREEDY_SIMPLIFY_REGION_LEVEL_DISABLED:
    cppLevel = aiir::GreedySimplifyRegionLevel::Disabled;
    break;
  case AIIR_GREEDY_SIMPLIFY_REGION_LEVEL_NORMAL:
    cppLevel = aiir::GreedySimplifyRegionLevel::Normal;
    break;
  case AIIR_GREEDY_SIMPLIFY_REGION_LEVEL_AGGRESSIVE:
    cppLevel = aiir::GreedySimplifyRegionLevel::Aggressive;
    break;
  }
  unwrap(config)->setRegionSimplificationLevel(cppLevel);
}

void aiirGreedyRewriteDriverConfigEnableConstantCSE(
    AiirGreedyRewriteDriverConfig config, bool enable) {
  unwrap(config)->enableConstantCSE(enable);
}

int64_t aiirGreedyRewriteDriverConfigGetMaxIterations(
    AiirGreedyRewriteDriverConfig config) {
  return unwrap(config)->getMaxIterations();
}

int64_t aiirGreedyRewriteDriverConfigGetMaxNumRewrites(
    AiirGreedyRewriteDriverConfig config) {
  return unwrap(config)->getMaxNumRewrites();
}

bool aiirGreedyRewriteDriverConfigGetUseTopDownTraversal(
    AiirGreedyRewriteDriverConfig config) {
  return unwrap(config)->getUseTopDownTraversal();
}

bool aiirGreedyRewriteDriverConfigIsFoldingEnabled(
    AiirGreedyRewriteDriverConfig config) {
  return unwrap(config)->isFoldingEnabled();
}

AiirGreedyRewriteStrictness aiirGreedyRewriteDriverConfigGetStrictness(
    AiirGreedyRewriteDriverConfig config) {
  aiir::GreedyRewriteStrictness cppStrictness = unwrap(config)->getStrictness();
  switch (cppStrictness) {
  case aiir::GreedyRewriteStrictness::AnyOp:
    return AIIR_GREEDY_REWRITE_STRICTNESS_ANY_OP;
  case aiir::GreedyRewriteStrictness::ExistingAndNewOps:
    return AIIR_GREEDY_REWRITE_STRICTNESS_EXISTING_AND_NEW_OPS;
  case aiir::GreedyRewriteStrictness::ExistingOps:
    return AIIR_GREEDY_REWRITE_STRICTNESS_EXISTING_OPS;
  }
  llvm_unreachable("Unknown GreedyRewriteStrictness");
}

AiirGreedySimplifyRegionLevel
aiirGreedyRewriteDriverConfigGetRegionSimplificationLevel(
    AiirGreedyRewriteDriverConfig config) {
  aiir::GreedySimplifyRegionLevel cppLevel =
      unwrap(config)->getRegionSimplificationLevel();
  switch (cppLevel) {
  case aiir::GreedySimplifyRegionLevel::Disabled:
    return AIIR_GREEDY_SIMPLIFY_REGION_LEVEL_DISABLED;
  case aiir::GreedySimplifyRegionLevel::Normal:
    return AIIR_GREEDY_SIMPLIFY_REGION_LEVEL_NORMAL;
  case aiir::GreedySimplifyRegionLevel::Aggressive:
    return AIIR_GREEDY_SIMPLIFY_REGION_LEVEL_AGGRESSIVE;
  }
  llvm_unreachable("Unknown GreedySimplifyRegionLevel");
}

bool aiirGreedyRewriteDriverConfigIsConstantCSEEnabled(
    AiirGreedyRewriteDriverConfig config) {
  return unwrap(config)->isConstantCSEEnabled();
}

AiirLogicalResult
aiirApplyPatternsAndFoldGreedily(AiirModule op,
                                 AiirFrozenRewritePatternSet patterns,
                                 AiirGreedyRewriteDriverConfig config) {
  return wrap(aiir::applyPatternsGreedily(unwrap(op), *unwrap(patterns),
                                          *unwrap(config)));
}

AiirLogicalResult
aiirApplyPatternsAndFoldGreedilyWithOp(AiirOperation op,
                                       AiirFrozenRewritePatternSet patterns,
                                       AiirGreedyRewriteDriverConfig config) {
  return wrap(aiir::applyPatternsGreedily(unwrap(op), *unwrap(patterns),
                                          *unwrap(config)));
}

void aiirWalkAndApplyPatterns(AiirOperation op,
                              AiirFrozenRewritePatternSet patterns) {
  aiir::walkAndApplyPatterns(unwrap(op), *unwrap(patterns));
}

AiirLogicalResult
aiirApplyPartialConversion(AiirOperation op, AiirConversionTarget target,
                           AiirFrozenRewritePatternSet patterns,
                           AiirConversionConfig config) {
  return wrap(aiir::applyPartialConversion(unwrap(op), *unwrap(target),
                                           *unwrap(patterns), *unwrap(config)));
}

AiirLogicalResult aiirApplyFullConversion(AiirOperation op,
                                          AiirConversionTarget target,
                                          AiirFrozenRewritePatternSet patterns,
                                          AiirConversionConfig config) {
  return wrap(aiir::applyFullConversion(unwrap(op), *unwrap(target),
                                        *unwrap(patterns), *unwrap(config)));
}

//===----------------------------------------------------------------------===//
/// ConversionConfig API
//===----------------------------------------------------------------------===//

AiirConversionConfig aiirConversionConfigCreate(void) {
  return wrap(new aiir::ConversionConfig());
}

void aiirConversionConfigDestroy(AiirConversionConfig config) {
  delete unwrap(config);
}

void aiirConversionConfigSetFoldingMode(AiirConversionConfig config,
                                        AiirDialectConversionFoldingMode mode) {
  aiir::DialectConversionFoldingMode cppMode;
  switch (mode) {
  case AIIR_DIALECT_CONVERSION_FOLDING_MODE_NEVER:
    cppMode = aiir::DialectConversionFoldingMode::Never;
    break;
  case AIIR_DIALECT_CONVERSION_FOLDING_MODE_BEFORE_PATTERNS:
    cppMode = aiir::DialectConversionFoldingMode::BeforePatterns;
    break;
  case AIIR_DIALECT_CONVERSION_FOLDING_MODE_AFTER_PATTERNS:
    cppMode = aiir::DialectConversionFoldingMode::AfterPatterns;
    break;
  }
  unwrap(config)->foldingMode = cppMode;
}

AiirDialectConversionFoldingMode
aiirConversionConfigGetFoldingMode(AiirConversionConfig config) {
  switch (unwrap(config)->foldingMode) {
  case aiir::DialectConversionFoldingMode::Never:
    return AIIR_DIALECT_CONVERSION_FOLDING_MODE_NEVER;
  case aiir::DialectConversionFoldingMode::BeforePatterns:
    return AIIR_DIALECT_CONVERSION_FOLDING_MODE_BEFORE_PATTERNS;
  case aiir::DialectConversionFoldingMode::AfterPatterns:
    return AIIR_DIALECT_CONVERSION_FOLDING_MODE_AFTER_PATTERNS;
  }
}

void aiirConversionConfigEnableBuildMaterializations(
    AiirConversionConfig config, bool enable) {
  unwrap(config)->buildMaterializations = enable;
}

bool aiirConversionConfigIsBuildMaterializationsEnabled(
    AiirConversionConfig config) {
  return unwrap(config)->buildMaterializations;
}

//===----------------------------------------------------------------------===//
/// PatternRewriter API
//===----------------------------------------------------------------------===//

AiirRewriterBase aiirPatternRewriterAsBase(AiirPatternRewriter rewriter) {
  return wrap(static_cast<aiir::RewriterBase *>(unwrap(rewriter)));
}

//===----------------------------------------------------------------------===//
/// ConversionPatternRewriter API
//===----------------------------------------------------------------------===//

AiirPatternRewriter aiirConversionPatternRewriterAsPatternRewriter(
    AiirConversionPatternRewriter rewriter) {
  return wrap(static_cast<aiir::PatternRewriter *>(unwrap(rewriter)));
}

AiirLogicalResult aiirConversionPatternRewriterConvertRegionTypes(
    AiirConversionPatternRewriter rewriter, AiirRegion region,
    AiirTypeConverter typeConverter) {
  return wrap(unwrap(rewriter)->convertRegionTypes(unwrap(region),
                                                   *unwrap(typeConverter)));
}

//===----------------------------------------------------------------------===//
/// ConversionTarget API
//===----------------------------------------------------------------------===//

AiirConversionTarget aiirConversionTargetCreate(AiirContext context) {
  return wrap(new aiir::ConversionTarget(*unwrap(context)));
}

void aiirConversionTargetDestroy(AiirConversionTarget target) {
  delete unwrap(target);
}

void aiirConversionTargetAddLegalOp(AiirConversionTarget target,
                                    AiirStringRef opName) {
  unwrap(target)->addLegalOp(
      aiir::OperationName(unwrap(opName), &unwrap(target)->getContext()));
}

void aiirConversionTargetAddIllegalOp(AiirConversionTarget target,
                                      AiirStringRef opName) {
  unwrap(target)->addIllegalOp(
      aiir::OperationName(unwrap(opName), &unwrap(target)->getContext()));
}

void aiirConversionTargetAddLegalDialect(AiirConversionTarget target,
                                         AiirStringRef dialectName) {
  unwrap(target)->addLegalDialect(unwrap(dialectName));
}

void aiirConversionTargetAddIllegalDialect(AiirConversionTarget target,
                                           AiirStringRef dialectName) {
  unwrap(target)->addIllegalDialect(unwrap(dialectName));
}

//===----------------------------------------------------------------------===//
/// TypeConverter API
//===----------------------------------------------------------------------===//

AiirTypeConverter aiirTypeConverterCreate() {
  return wrap(new aiir::TypeConverter());
}

void aiirTypeConverterDestroy(AiirTypeConverter typeConverter) {
  delete unwrap(typeConverter);
}

void aiirTypeConverterAddConversion(
    AiirTypeConverter typeConverter,
    AiirTypeConverterConversionCallback convertType, void *userData) {
  unwrap(typeConverter)
      ->addConversion(
          [convertType, userData](Type type) -> std::optional<Type> {
            AiirType converted{nullptr};
            AiirLogicalResult result =
                convertType(wrap(type), &converted, userData);
            if (aiirLogicalResultIsFailure(result))
              return std::nullopt; // allowed to try another conversion function
            if (aiirTypeIsNull(converted))
              return nullptr;
            return unwrap(converted);
          });
}

AiirType aiirTypeConverterConvertType(AiirTypeConverter typeConverter,
                                      AiirType type) {
  return wrap(unwrap(typeConverter)->convertType(unwrap(type)));
}

//===----------------------------------------------------------------------===//
/// ConversionPattern API
//===----------------------------------------------------------------------===//

namespace aiir {

class ExternalConversionPattern : public aiir::ConversionPattern {
public:
  ExternalConversionPattern(AiirConversionPatternCallbacks callbacks,
                            void *userData, StringRef rootName,
                            PatternBenefit benefit, AIIRContext *context,
                            TypeConverter *typeConverter,
                            ArrayRef<StringRef> generatedNames)
      : ConversionPattern(*typeConverter, rootName, benefit, context,
                          generatedNames),
        callbacks(callbacks), userData(userData) {
    if (callbacks.construct)
      callbacks.construct(userData);
  }

  ~ExternalConversionPattern() {
    if (callbacks.destruct)
      callbacks.destruct(userData);
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    std::vector<AiirValue> wrappedOperands;
    for (Value val : operands)
      wrappedOperands.push_back(wrap(val));
    return unwrap(callbacks.matchAndRewrite(
        wrap(static_cast<const aiir::ConversionPattern *>(this)), wrap(op),
        wrappedOperands.size(), wrappedOperands.data(), wrap(&rewriter),
        userData));
  }

private:
  AiirConversionPatternCallbacks callbacks;
  void *userData;
};

} // namespace aiir

AiirConversionPattern aiirOpConversionPatternCreate(
    AiirStringRef rootName, unsigned benefit, AiirContext context,
    AiirTypeConverter typeConverter, AiirConversionPatternCallbacks callbacks,
    void *userData, size_t nGeneratedNames, AiirStringRef *generatedNames) {
  std::vector<aiir::StringRef> generatedNamesVec;
  generatedNamesVec.reserve(nGeneratedNames);
  for (size_t i = 0; i < nGeneratedNames; ++i)
    generatedNamesVec.push_back(unwrap(generatedNames[i]));
  return wrap(new aiir::ExternalConversionPattern(
      callbacks, userData, unwrap(rootName), PatternBenefit(benefit),
      unwrap(context), unwrap(typeConverter), generatedNamesVec));
}

AiirTypeConverter
aiirConversionPatternGetTypeConverter(AiirConversionPattern pattern) {
  return wrap(const_cast<TypeConverter *>(unwrap(pattern)->getTypeConverter()));
}

AiirRewritePattern
aiirConversionPatternAsRewritePattern(AiirConversionPattern pattern) {
  return wrap(static_cast<const RewritePattern *>(unwrap(pattern)));
}

//===----------------------------------------------------------------------===//
/// RewritePattern API
//===----------------------------------------------------------------------===//

namespace aiir {

class ExternalRewritePattern : public aiir::RewritePattern {
public:
  ExternalRewritePattern(AiirRewritePatternCallbacks callbacks, void *userData,
                         StringRef rootName, PatternBenefit benefit,
                         AIIRContext *context,
                         ArrayRef<StringRef> generatedNames)
      : RewritePattern(rootName, benefit, context, generatedNames),
        callbacks(callbacks), userData(userData) {
    if (callbacks.construct)
      callbacks.construct(userData);
  }

  ~ExternalRewritePattern() {
    if (callbacks.destruct)
      callbacks.destruct(userData);
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return unwrap(callbacks.matchAndRewrite(
        wrap(static_cast<const aiir::RewritePattern *>(this)), wrap(op),
        wrap(&rewriter), userData));
  }

private:
  AiirRewritePatternCallbacks callbacks;
  void *userData;
};

} // namespace aiir

AiirRewritePattern aiirOpRewritePatternCreate(
    AiirStringRef rootName, unsigned benefit, AiirContext context,
    AiirRewritePatternCallbacks callbacks, void *userData,
    size_t nGeneratedNames, AiirStringRef *generatedNames) {
  std::vector<aiir::StringRef> generatedNamesVec;
  generatedNamesVec.reserve(nGeneratedNames);
  for (size_t i = 0; i < nGeneratedNames; ++i) {
    generatedNamesVec.push_back(unwrap(generatedNames[i]));
  }
  return wrap(new aiir::ExternalRewritePattern(
      callbacks, userData, unwrap(rootName), PatternBenefit(benefit),
      unwrap(context), generatedNamesVec));
}

//===----------------------------------------------------------------------===//
/// RewritePatternSet API
//===----------------------------------------------------------------------===//

AiirRewritePatternSet aiirRewritePatternSetCreate(AiirContext context) {
  return wrap(new aiir::RewritePatternSet(unwrap(context)));
}

AiirContext aiirRewritePatternSetGetContext(AiirRewritePatternSet set) {
  return wrap(unwrap(set)->getContext());
}

void aiirRewritePatternSetDestroy(AiirRewritePatternSet set) {
  delete unwrap(set);
}

void aiirRewritePatternSetAdd(AiirRewritePatternSet set,
                              AiirRewritePattern pattern) {
  std::unique_ptr<aiir::RewritePattern> patternPtr(
      const_cast<aiir::RewritePattern *>(unwrap(pattern)));
  pattern.ptr = nullptr;
  unwrap(set)->add(std::move(patternPtr));
}

//===----------------------------------------------------------------------===//
/// PDLPatternModule API
//===----------------------------------------------------------------------===//

#if AIIR_ENABLE_PDL_IN_PATTERNMATCH
AiirPDLPatternModule aiirPDLPatternModuleFromModule(AiirModule op) {
  return wrap(new aiir::PDLPatternModule(
      aiir::OwningOpRef<aiir::ModuleOp>(unwrap(op))));
}

void aiirPDLPatternModuleDestroy(AiirPDLPatternModule op) {
  delete unwrap(op);
  op.ptr = nullptr;
}

AiirRewritePatternSet
aiirRewritePatternSetFromPDLPatternModule(AiirPDLPatternModule op) {
  auto *m = new aiir::RewritePatternSet(std::move(*unwrap(op)));
  op.ptr = nullptr;
  return wrap(m);
}

AiirValue aiirPDLValueAsValue(AiirPDLValue value) {
  return wrap(unwrap(value)->dyn_cast<aiir::Value>());
}

AiirType aiirPDLValueAsType(AiirPDLValue value) {
  return wrap(unwrap(value)->dyn_cast<aiir::Type>());
}

AiirOperation aiirPDLValueAsOperation(AiirPDLValue value) {
  return wrap(unwrap(value)->dyn_cast<aiir::Operation *>());
}

AiirAttribute aiirPDLValueAsAttribute(AiirPDLValue value) {
  return wrap(unwrap(value)->dyn_cast<aiir::Attribute>());
}

void aiirPDLResultListPushBackValue(AiirPDLResultList results,
                                    AiirValue value) {
  unwrap(results)->push_back(unwrap(value));
}

void aiirPDLResultListPushBackType(AiirPDLResultList results, AiirType value) {
  unwrap(results)->push_back(unwrap(value));
}

void aiirPDLResultListPushBackOperation(AiirPDLResultList results,
                                        AiirOperation value) {
  unwrap(results)->push_back(unwrap(value));
}

void aiirPDLResultListPushBackAttribute(AiirPDLResultList results,
                                        AiirAttribute value) {
  unwrap(results)->push_back(unwrap(value));
}

inline std::vector<AiirPDLValue> wrap(ArrayRef<PDLValue> values) {
  std::vector<AiirPDLValue> aiirValues;
  aiirValues.reserve(values.size());
  for (auto &value : values) {
    aiirValues.push_back(wrap(&value));
  }
  return aiirValues;
}

void aiirPDLPatternModuleRegisterRewriteFunction(
    AiirPDLPatternModule pdlModule, AiirStringRef name,
    AiirPDLRewriteFunction rewriteFn, void *userData) {
  unwrap(pdlModule)->registerRewriteFunction(
      unwrap(name),
      [userData, rewriteFn](PatternRewriter &rewriter, PDLResultList &results,
                            ArrayRef<PDLValue> values) -> LogicalResult {
        std::vector<AiirPDLValue> aiirValues = wrap(values);
        return unwrap(rewriteFn(wrap(&rewriter), wrap(&results),
                                aiirValues.size(), aiirValues.data(),
                                userData));
      });
}

void aiirPDLPatternModuleRegisterConstraintFunction(
    AiirPDLPatternModule pdlModule, AiirStringRef name,
    AiirPDLConstraintFunction constraintFn, void *userData) {
  unwrap(pdlModule)->registerConstraintFunction(
      unwrap(name),
      [userData, constraintFn](PatternRewriter &rewriter,
                               PDLResultList &results,
                               ArrayRef<PDLValue> values) -> LogicalResult {
        std::vector<AiirPDLValue> aiirValues = wrap(values);
        return unwrap(constraintFn(wrap(&rewriter), wrap(&results),
                                   aiirValues.size(), aiirValues.data(),
                                   userData));
      });
}
#endif // AIIR_ENABLE_PDL_IN_PATTERNMATCH
