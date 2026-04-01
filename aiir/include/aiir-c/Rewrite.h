//===-- aiir-c/Rewrite.h - Helpers for C API to Rewrites ----------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the registration and creation method for
// rewrite patterns.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_REWRITE_H
#define AIIR_C_REWRITE_H

#include "aiir-c/IR.h"
#include "aiir-c/Support.h"
#include "aiir/Config/aiir-config.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
/// Opaque type declarations (see aiir-c/IR.h for more details).
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(AiirRewriterBase, void);
DEFINE_C_API_STRUCT(AiirFrozenRewritePatternSet, void);
DEFINE_C_API_STRUCT(AiirGreedyRewriteDriverConfig, void);

/// Greedy rewrite strictness levels.
typedef enum {
  /// No restrictions wrt. which ops are processed.
  AIIR_GREEDY_REWRITE_STRICTNESS_ANY_OP,
  /// Only pre-existing and newly created ops are processed.
  AIIR_GREEDY_REWRITE_STRICTNESS_EXISTING_AND_NEW_OPS,
  /// Only pre-existing ops are processed.
  AIIR_GREEDY_REWRITE_STRICTNESS_EXISTING_OPS
} AiirGreedyRewriteStrictness;

/// Greedy simplify region levels.
typedef enum {
  /// Disable region control-flow simplification.
  AIIR_GREEDY_SIMPLIFY_REGION_LEVEL_DISABLED,
  /// Run the normal simplification (e.g. dead args elimination).
  AIIR_GREEDY_SIMPLIFY_REGION_LEVEL_NORMAL,
  /// Run extra simplifications (e.g. block merging).
  AIIR_GREEDY_SIMPLIFY_REGION_LEVEL_AGGRESSIVE
} AiirGreedySimplifyRegionLevel;
DEFINE_C_API_STRUCT(AiirRewritePatternSet, void);
DEFINE_C_API_STRUCT(AiirPatternRewriter, void);
DEFINE_C_API_STRUCT(AiirRewritePattern, const void);
DEFINE_C_API_STRUCT(AiirConversionTarget, void);
DEFINE_C_API_STRUCT(AiirConversionPattern, const void);
DEFINE_C_API_STRUCT(AiirTypeConverter, void);
DEFINE_C_API_STRUCT(AiirConversionPatternRewriter, void);
DEFINE_C_API_STRUCT(AiirConversionConfig, void);

//===----------------------------------------------------------------------===//
/// RewriterBase API inherited from OpBuilder
//===----------------------------------------------------------------------===//

/// Get the AIIR context referenced by the rewriter.
AIIR_CAPI_EXPORTED AiirContext
aiirRewriterBaseGetContext(AiirRewriterBase rewriter);

//===----------------------------------------------------------------------===//
/// Insertion points methods
//===----------------------------------------------------------------------===//

// These do not include functions using Block::iterator or Region::iterator, as
// they are not exposed by the C API yet. Similarly for methods using
// `InsertPoint` directly.

/// Reset the insertion point to no location.  Creating an operation without a
/// set insertion point is an error, but this can still be useful when the
/// current insertion point a builder refers to is being removed.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseClearInsertionPoint(AiirRewriterBase rewriter);

/// Sets the insertion point to the specified operation, which will cause
/// subsequent insertions to go right before it.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseSetInsertionPointBefore(AiirRewriterBase rewriter,
                                        AiirOperation op);

/// Sets the insertion point to the node after the specified operation, which
/// will cause subsequent insertions to go right after it.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseSetInsertionPointAfter(AiirRewriterBase rewriter,
                                       AiirOperation op);

/// Sets the insertion point to the node after the specified value. If value
/// has a defining operation, sets the insertion point to the node after such
/// defining operation. This will cause subsequent insertions to go right
/// after it. Otherwise, value is a BlockArgument. Sets the insertion point to
/// the start of its block.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseSetInsertionPointAfterValue(AiirRewriterBase rewriter,
                                            AiirValue value);

/// Sets the insertion point to the start of the specified block.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseSetInsertionPointToStart(AiirRewriterBase rewriter,
                                         AiirBlock block);

/// Sets the insertion point to the end of the specified block.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseSetInsertionPointToEnd(AiirRewriterBase rewriter,
                                       AiirBlock block);

/// Return the block the current insertion point belongs to.  Note that the
/// insertion point is not necessarily the end of the block.
AIIR_CAPI_EXPORTED AiirBlock
aiirRewriterBaseGetInsertionBlock(AiirRewriterBase rewriter);

/// Returns the current block of the rewriter.
AIIR_CAPI_EXPORTED AiirBlock
aiirRewriterBaseGetBlock(AiirRewriterBase rewriter);

/// Returns the operation right after the current insertion point
/// of the rewriter. A null AiirOperation will be returned
// if the current insertion point is at the end of the block.
AIIR_CAPI_EXPORTED AiirOperation
aiirRewriterBaseGetOperationAfterInsertion(AiirRewriterBase rewriter);

//===----------------------------------------------------------------------===//
/// Block and operation creation/insertion/cloning
//===----------------------------------------------------------------------===//

// These functions do not include the IRMapper, as it is not yet exposed by the
// C API.

/// Add new block with 'argTypes' arguments and set the insertion point to the
/// end of it. The block is placed before 'insertBefore'. `locs` contains the
/// locations of the inserted arguments, and should match the size of
/// `argTypes`.
AIIR_CAPI_EXPORTED AiirBlock aiirRewriterBaseCreateBlockBefore(
    AiirRewriterBase rewriter, AiirBlock insertBefore, intptr_t nArgTypes,
    AiirType const *argTypes, AiirLocation const *locations);

/// Insert the given operation at the current insertion point and return it.
AIIR_CAPI_EXPORTED AiirOperation
aiirRewriterBaseInsert(AiirRewriterBase rewriter, AiirOperation op);

/// Creates a deep copy of the specified operation.
AIIR_CAPI_EXPORTED AiirOperation
aiirRewriterBaseClone(AiirRewriterBase rewriter, AiirOperation op);

/// Creates a deep copy of this operation but keep the operation regions
/// empty.
AIIR_CAPI_EXPORTED AiirOperation aiirRewriterBaseCloneWithoutRegions(
    AiirRewriterBase rewriter, AiirOperation op);

/// Clone the blocks that belong to "region" before the given position in
/// another region "parent".
AIIR_CAPI_EXPORTED void
aiirRewriterBaseCloneRegionBefore(AiirRewriterBase rewriter, AiirRegion region,
                                  AiirBlock before);

//===----------------------------------------------------------------------===//
/// RewriterBase API
//===----------------------------------------------------------------------===//

/// Move the blocks that belong to "region" before the given position in
/// another region "parent". The two regions must be different. The caller
/// is responsible for creating or updating the operation transferring flow
/// of control to the region and passing it the correct block arguments.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseInlineRegionBefore(AiirRewriterBase rewriter, AiirRegion region,
                                   AiirBlock before);

/// Replace the results of the given (original) operation with the specified
/// list of values (replacements). The result types of the given op and the
/// replacements must match. The original op is erased.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseReplaceOpWithValues(AiirRewriterBase rewriter, AiirOperation op,
                                    intptr_t nValues, AiirValue const *values);

/// Replace the results of the given (original) operation with the specified
/// new op (replacement). The result types of the two ops must match. The
/// original op is erased.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseReplaceOpWithOperation(AiirRewriterBase rewriter,
                                       AiirOperation op, AiirOperation newOp);

/// Erases an operation that is known to have no uses.
AIIR_CAPI_EXPORTED void aiirRewriterBaseEraseOp(AiirRewriterBase rewriter,
                                                AiirOperation op);

/// Erases a block along with all operations inside it.
AIIR_CAPI_EXPORTED void aiirRewriterBaseEraseBlock(AiirRewriterBase rewriter,
                                                   AiirBlock block);

/// Inline the operations of block 'source' before the operation 'op'. The
/// source block will be deleted and must have no uses. 'argValues' is used to
/// replace the block arguments of 'source'
///
/// The source block must have no successors. Otherwise, the resulting IR
/// would have unreachable operations.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseInlineBlockBefore(AiirRewriterBase rewriter, AiirBlock source,
                                  AiirOperation op, intptr_t nArgValues,
                                  AiirValue const *argValues);

/// Inline the operations of block 'source' into the end of block 'dest'. The
/// source block will be deleted and must have no uses. 'argValues' is used to
/// replace the block arguments of 'source'
///
/// The dest block must have no successors. Otherwise, the resulting IR would
/// have unreachable operation.
AIIR_CAPI_EXPORTED void aiirRewriterBaseMergeBlocks(AiirRewriterBase rewriter,
                                                    AiirBlock source,
                                                    AiirBlock dest,
                                                    intptr_t nArgValues,
                                                    AiirValue const *argValues);

/// Unlink this operation from its current block and insert it right before
/// `existingOp` which may be in the same or another block in the same
/// function.
AIIR_CAPI_EXPORTED void aiirRewriterBaseMoveOpBefore(AiirRewriterBase rewriter,
                                                     AiirOperation op,
                                                     AiirOperation existingOp);

/// Unlink this operation from its current block and insert it right after
/// `existingOp` which may be in the same or another block in the same
/// function.
AIIR_CAPI_EXPORTED void aiirRewriterBaseMoveOpAfter(AiirRewriterBase rewriter,
                                                    AiirOperation op,
                                                    AiirOperation existingOp);

/// Unlink this block and insert it right before `existingBlock`.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseMoveBlockBefore(AiirRewriterBase rewriter, AiirBlock block,
                                AiirBlock existingBlock);

/// This method is used to notify the rewriter that an in-place operation
/// modification is about to happen. A call to this function *must* be
/// followed by a call to either `finalizeOpModification` or
/// `cancelOpModification`. This is a minor efficiency win (it avoids creating
/// a new operation and removing the old one) but also often allows simpler
/// code in the client.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseStartOpModification(AiirRewriterBase rewriter,
                                    AiirOperation op);

/// This method is used to signal the end of an in-place modification of the
/// given operation. This can only be called on operations that were provided
/// to a call to `startOpModification`.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseFinalizeOpModification(AiirRewriterBase rewriter,
                                       AiirOperation op);

/// This method cancels a pending in-place modification. This can only be
/// called on operations that were provided to a call to
/// `startOpModification`.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseCancelOpModification(AiirRewriterBase rewriter,
                                     AiirOperation op);

/// Find uses of `from` and replace them with `to`. Also notify the listener
/// about every in-place op modification (for every use that was replaced).
AIIR_CAPI_EXPORTED void
aiirRewriterBaseReplaceAllUsesWith(AiirRewriterBase rewriter, AiirValue from,
                                   AiirValue to);

/// Find uses of `from` and replace them with `to`. Also notify the listener
/// about every in-place op modification (for every use that was replaced).
AIIR_CAPI_EXPORTED void aiirRewriterBaseReplaceAllValueRangeUsesWith(
    AiirRewriterBase rewriter, intptr_t nValues, AiirValue const *from,
    AiirValue const *to);

/// Find uses of `from` and replace them with `to`. Also notify the listener
/// about every in-place op modification (for every use that was replaced)
/// and that the `from` operation is about to be replaced.
AIIR_CAPI_EXPORTED void
aiirRewriterBaseReplaceAllOpUsesWithValueRange(AiirRewriterBase rewriter,
                                               AiirOperation from, intptr_t nTo,
                                               AiirValue const *to);

/// Find uses of `from` and replace them with `to`. Also notify the listener
/// about every in-place op modification (for every use that was replaced)
/// and that the `from` operation is about to be replaced.
AIIR_CAPI_EXPORTED void aiirRewriterBaseReplaceAllOpUsesWithOperation(
    AiirRewriterBase rewriter, AiirOperation from, AiirOperation to);

/// Find uses of `from` within `block` and replace them with `to`. Also notify
/// the listener about every in-place op modification (for every use that was
/// replaced). The optional `allUsesReplaced` flag is set to "true" if all
/// uses were replaced.
AIIR_CAPI_EXPORTED void aiirRewriterBaseReplaceOpUsesWithinBlock(
    AiirRewriterBase rewriter, AiirOperation op, intptr_t nNewValues,
    AiirValue const *newValues, AiirBlock block);

/// Find uses of `from` and replace them with `to` except if the user is
/// `exceptedUser`. Also notify the listener about every in-place op
/// modification (for every use that was replaced).
AIIR_CAPI_EXPORTED void
aiirRewriterBaseReplaceAllUsesExcept(AiirRewriterBase rewriter, AiirValue from,
                                     AiirValue to, AiirOperation exceptedUser);

//===----------------------------------------------------------------------===//
/// IRRewriter API
//===----------------------------------------------------------------------===//

/// Create an IRRewriter and transfer ownership to the caller.
AIIR_CAPI_EXPORTED AiirRewriterBase aiirIRRewriterCreate(AiirContext context);

/// Create an IRRewriter and transfer ownership to the caller. Additionally
/// set the insertion point before the operation.
AIIR_CAPI_EXPORTED AiirRewriterBase
aiirIRRewriterCreateFromOp(AiirOperation op);

/// Takes an IRRewriter owned by the caller and destroys it. It is the
/// responsibility of the user to only pass an IRRewriter class.
AIIR_CAPI_EXPORTED void aiirIRRewriterDestroy(AiirRewriterBase rewriter);

//===----------------------------------------------------------------------===//
/// FrozenRewritePatternSet API
//===----------------------------------------------------------------------===//

/// Freeze the given AiirRewritePatternSet to a AiirFrozenRewritePatternSet.
/// Note that the ownership of the input set is transferred into the frozen set
/// after this call.
AIIR_CAPI_EXPORTED AiirFrozenRewritePatternSet
aiirFreezeRewritePattern(AiirRewritePatternSet set);

/// Destroy the given AiirFrozenRewritePatternSet.
AIIR_CAPI_EXPORTED void
aiirFrozenRewritePatternSetDestroy(AiirFrozenRewritePatternSet set);

AIIR_CAPI_EXPORTED AiirLogicalResult aiirApplyPatternsAndFoldGreedilyWithOp(
    AiirOperation op, AiirFrozenRewritePatternSet patterns,
    AiirGreedyRewriteDriverConfig);

AIIR_CAPI_EXPORTED AiirLogicalResult aiirApplyPatternsAndFoldGreedily(
    AiirModule op, AiirFrozenRewritePatternSet patterns,
    AiirGreedyRewriteDriverConfig config);

//===----------------------------------------------------------------------===//
/// GreedyRewriteDriverConfig API
//===----------------------------------------------------------------------===//

/// Creates a greedy rewrite driver configuration with default settings.
AIIR_CAPI_EXPORTED AiirGreedyRewriteDriverConfig
aiirGreedyRewriteDriverConfigCreate(void);

/// Destroys a greedy rewrite driver configuration.
AIIR_CAPI_EXPORTED void
aiirGreedyRewriteDriverConfigDestroy(AiirGreedyRewriteDriverConfig config);

/// Sets the maximum number of iterations for the greedy rewrite driver.
/// Use -1 for no limit.
AIIR_CAPI_EXPORTED void aiirGreedyRewriteDriverConfigSetMaxIterations(
    AiirGreedyRewriteDriverConfig config, int64_t maxIterations);

/// Sets the maximum number of rewrites within an iteration.
/// Use -1 for no limit.
AIIR_CAPI_EXPORTED void aiirGreedyRewriteDriverConfigSetMaxNumRewrites(
    AiirGreedyRewriteDriverConfig config, int64_t maxNumRewrites);

/// Sets whether to use top-down traversal for the initial population of the
/// worklist.
AIIR_CAPI_EXPORTED void aiirGreedyRewriteDriverConfigSetUseTopDownTraversal(
    AiirGreedyRewriteDriverConfig config, bool useTopDownTraversal);

/// Enables or disables folding during greedy rewriting.
AIIR_CAPI_EXPORTED void
aiirGreedyRewriteDriverConfigEnableFolding(AiirGreedyRewriteDriverConfig config,
                                           bool enable);

/// Sets the strictness level for the greedy rewrite driver.
AIIR_CAPI_EXPORTED void aiirGreedyRewriteDriverConfigSetStrictness(
    AiirGreedyRewriteDriverConfig config,
    AiirGreedyRewriteStrictness strictness);

/// Sets the region simplification level.
AIIR_CAPI_EXPORTED void
aiirGreedyRewriteDriverConfigSetRegionSimplificationLevel(
    AiirGreedyRewriteDriverConfig config, AiirGreedySimplifyRegionLevel level);

/// Enables or disables constant CSE.
AIIR_CAPI_EXPORTED void aiirGreedyRewriteDriverConfigEnableConstantCSE(
    AiirGreedyRewriteDriverConfig config, bool enable);

/// Gets the maximum number of iterations for the greedy rewrite driver.
AIIR_CAPI_EXPORTED int64_t aiirGreedyRewriteDriverConfigGetMaxIterations(
    AiirGreedyRewriteDriverConfig config);

/// Gets the maximum number of rewrites within an iteration.
AIIR_CAPI_EXPORTED int64_t aiirGreedyRewriteDriverConfigGetMaxNumRewrites(
    AiirGreedyRewriteDriverConfig config);

/// Gets whether top-down traversal is used for initial worklist population.
AIIR_CAPI_EXPORTED bool aiirGreedyRewriteDriverConfigGetUseTopDownTraversal(
    AiirGreedyRewriteDriverConfig config);

/// Gets whether folding is enabled during greedy rewriting.
AIIR_CAPI_EXPORTED bool aiirGreedyRewriteDriverConfigIsFoldingEnabled(
    AiirGreedyRewriteDriverConfig config);

/// Gets the strictness level for the greedy rewrite driver.
AIIR_CAPI_EXPORTED AiirGreedyRewriteStrictness
aiirGreedyRewriteDriverConfigGetStrictness(
    AiirGreedyRewriteDriverConfig config);

/// Gets the region simplification level.
AIIR_CAPI_EXPORTED AiirGreedySimplifyRegionLevel
aiirGreedyRewriteDriverConfigGetRegionSimplificationLevel(
    AiirGreedyRewriteDriverConfig config);

/// Gets whether constant CSE is enabled.
AIIR_CAPI_EXPORTED bool aiirGreedyRewriteDriverConfigIsConstantCSEEnabled(
    AiirGreedyRewriteDriverConfig config);

/// Applies the given patterns to the given op by a fast walk-based pattern
/// rewrite driver.
AIIR_CAPI_EXPORTED void
aiirWalkAndApplyPatterns(AiirOperation op,
                         AiirFrozenRewritePatternSet patterns);

/// Apply a partial conversion on the given operation.
AIIR_CAPI_EXPORTED AiirLogicalResult aiirApplyPartialConversion(
    AiirOperation op, AiirConversionTarget target,
    AiirFrozenRewritePatternSet patterns, AiirConversionConfig config);

/// Apply a full conversion on the given operation.
AIIR_CAPI_EXPORTED AiirLogicalResult aiirApplyFullConversion(
    AiirOperation op, AiirConversionTarget target,
    AiirFrozenRewritePatternSet patterns, AiirConversionConfig config);

//===----------------------------------------------------------------------===//
/// ConversionConfig API
//===----------------------------------------------------------------------===//

/// Create a default ConversionConfig.
AIIR_CAPI_EXPORTED AiirConversionConfig aiirConversionConfigCreate(void);

/// Destroy the given ConversionConfig.
AIIR_CAPI_EXPORTED void
aiirConversionConfigDestroy(AiirConversionConfig config);

typedef enum {
  AIIR_DIALECT_CONVERSION_FOLDING_MODE_NEVER,
  AIIR_DIALECT_CONVERSION_FOLDING_MODE_BEFORE_PATTERNS,
  AIIR_DIALECT_CONVERSION_FOLDING_MODE_AFTER_PATTERNS,
} AiirDialectConversionFoldingMode;

/// Set the folding mode for the given ConversionConfig.
AIIR_CAPI_EXPORTED void
aiirConversionConfigSetFoldingMode(AiirConversionConfig config,
                                   AiirDialectConversionFoldingMode mode);

/// Get the folding mode for the given ConversionConfig.
AIIR_CAPI_EXPORTED AiirDialectConversionFoldingMode
aiirConversionConfigGetFoldingMode(AiirConversionConfig config);

/// Enable or disable building materializations during conversion.
AIIR_CAPI_EXPORTED void
aiirConversionConfigEnableBuildMaterializations(AiirConversionConfig config,
                                                bool enable);

/// Check if building materializations during conversion is enabled.
AIIR_CAPI_EXPORTED bool
aiirConversionConfigIsBuildMaterializationsEnabled(AiirConversionConfig config);

//===----------------------------------------------------------------------===//
/// PatternRewriter API
//===----------------------------------------------------------------------===//

/// Cast the PatternRewriter to a RewriterBase
AIIR_CAPI_EXPORTED AiirRewriterBase
aiirPatternRewriterAsBase(AiirPatternRewriter rewriter);

//===----------------------------------------------------------------------===//
/// ConversionPatternRewriter API
//===----------------------------------------------------------------------===//

/// Cast the ConversionPatternRewriter to a PatternRewriter
AIIR_CAPI_EXPORTED AiirPatternRewriter
aiirConversionPatternRewriterAsPatternRewriter(
    AiirConversionPatternRewriter rewriter);

/// Apply a signature conversion to each block in the given region.
AIIR_CAPI_EXPORTED AiirLogicalResult
aiirConversionPatternRewriterConvertRegionTypes(
    AiirConversionPatternRewriter rewriter, AiirRegion region,
    AiirTypeConverter typeConverter);

//===----------------------------------------------------------------------===//
/// ConversionTarget API
//===----------------------------------------------------------------------===//

/// Create an empty ConversionTarget.
AIIR_CAPI_EXPORTED AiirConversionTarget
aiirConversionTargetCreate(AiirContext context);

/// Destroy the given ConversionTarget.
AIIR_CAPI_EXPORTED void
aiirConversionTargetDestroy(AiirConversionTarget target);

/// Register the given operations as legal.
AIIR_CAPI_EXPORTED void
aiirConversionTargetAddLegalOp(AiirConversionTarget target,
                               AiirStringRef opName);

/// Register the given operations as illegal.
AIIR_CAPI_EXPORTED void
aiirConversionTargetAddIllegalOp(AiirConversionTarget target,
                                 AiirStringRef opName);

/// Register the operations of the given dialect as legal.
AIIR_CAPI_EXPORTED void
aiirConversionTargetAddLegalDialect(AiirConversionTarget target,
                                    AiirStringRef dialectName);

/// Register the operations of the given dialect as illegal.
AIIR_CAPI_EXPORTED void
aiirConversionTargetAddIllegalDialect(AiirConversionTarget target,
                                      AiirStringRef dialectName);

//===----------------------------------------------------------------------===//
/// TypeConverter API
//===----------------------------------------------------------------------===//

/// Create a TypeConverter.
AIIR_CAPI_EXPORTED AiirTypeConverter aiirTypeConverterCreate(void);

/// Destroy the given TypeConverter.
AIIR_CAPI_EXPORTED void
aiirTypeConverterDestroy(AiirTypeConverter typeConverter);

/// Callback type for type conversion functions.
/// Returns failure or sets convertedType to AiirType{NULL} to indicate failure.
/// If failure is returned, the converter is allowed to try another
/// conversion function to perform the conversion.
typedef AiirLogicalResult (*AiirTypeConverterConversionCallback)(
    AiirType type, AiirType *convertedType, void *userData);

/// Add a type conversion function to the given TypeConverter.
AIIR_CAPI_EXPORTED void
aiirTypeConverterAddConversion(AiirTypeConverter typeConverter,
                               AiirTypeConverterConversionCallback convertType,
                               void *userData);

/// Convert the given type using the given TypeConverter.
AIIR_CAPI_EXPORTED AiirType
aiirTypeConverterConvertType(AiirTypeConverter typeConverter, AiirType type);

//===----------------------------------------------------------------------===//
/// ConversionPattern API
//===----------------------------------------------------------------------===//

typedef struct {
  /// Optional constructor for the user data.
  /// Set to nullptr to disable it.
  void (*construct)(void *userData);
  /// Optional destructor for the user data.
  /// Set to nullptr to disable it.
  void (*destruct)(void *userData);
  /// The callback function to match against code rooted at the specified
  /// operation, and perform the conversion rewrite if the match is successful,
  /// corresponding to ConversionPattern::matchAndRewrite.
  AiirLogicalResult (*matchAndRewrite)(AiirConversionPattern pattern,
                                       AiirOperation op, intptr_t nOperands,
                                       AiirValue *operands,
                                       AiirConversionPatternRewriter rewriter,
                                       void *userData);
} AiirConversionPatternCallbacks;

/// Create a conversion pattern that matches the operation with the given
/// rootName, corresponding to aiir::OpConversionPattern.
AIIR_CAPI_EXPORTED AiirConversionPattern aiirOpConversionPatternCreate(
    AiirStringRef rootName, unsigned benefit, AiirContext context,
    AiirTypeConverter typeConverter, AiirConversionPatternCallbacks callbacks,
    void *userData, size_t nGeneratedNames, AiirStringRef *generatedNames);

/// Get the type converter used by this conversion pattern.
AIIR_CAPI_EXPORTED AiirTypeConverter
aiirConversionPatternGetTypeConverter(AiirConversionPattern pattern);

/// Cast the ConversionPattern to a RewritePattern.
AIIR_CAPI_EXPORTED AiirRewritePattern
aiirConversionPatternAsRewritePattern(AiirConversionPattern pattern);

//===----------------------------------------------------------------------===//
/// RewritePattern API
//===----------------------------------------------------------------------===//

/// Callbacks to construct a rewrite pattern.
typedef struct {
  /// Optional constructor for the user data.
  /// Set to nullptr to disable it.
  void (*construct)(void *userData);
  /// Optional destructor for the user data.
  /// Set to nullptr to disable it.
  void (*destruct)(void *userData);
  /// The callback function to match against code rooted at the specified
  /// operation, and perform the rewrite if the match is successful,
  /// corresponding to RewritePattern::matchAndRewrite.
  AiirLogicalResult (*matchAndRewrite)(AiirRewritePattern pattern,
                                       AiirOperation op,
                                       AiirPatternRewriter rewriter,
                                       void *userData);
} AiirRewritePatternCallbacks;

/// Create a rewrite pattern that matches the operation
/// with the given rootName, corresponding to aiir::OpRewritePattern.
AIIR_CAPI_EXPORTED AiirRewritePattern aiirOpRewritePatternCreate(
    AiirStringRef rootName, unsigned benefit, AiirContext context,
    AiirRewritePatternCallbacks callbacks, void *userData,
    size_t nGeneratedNames, AiirStringRef *generatedNames);

//===----------------------------------------------------------------------===//
/// RewritePatternSet API
//===----------------------------------------------------------------------===//

/// Create an empty AiirRewritePatternSet.
AIIR_CAPI_EXPORTED AiirRewritePatternSet
aiirRewritePatternSetCreate(AiirContext context);

/// Get the context associated with a AiirRewritePatternSet.
AIIR_CAPI_EXPORTED AiirContext
aiirRewritePatternSetGetContext(AiirRewritePatternSet set);

/// Destruct the given AiirRewritePatternSet.
AIIR_CAPI_EXPORTED void aiirRewritePatternSetDestroy(AiirRewritePatternSet set);

/// Add the given AiirRewritePattern into a AiirRewritePatternSet.
/// Note that the ownership of the pattern is transferred to the set after this
/// call.
AIIR_CAPI_EXPORTED void aiirRewritePatternSetAdd(AiirRewritePatternSet set,
                                                 AiirRewritePattern pattern);

//===----------------------------------------------------------------------===//
/// PDLPatternModule API
//===----------------------------------------------------------------------===//

#if AIIR_ENABLE_PDL_IN_PATTERNMATCH
DEFINE_C_API_STRUCT(AiirPDLPatternModule, void);
DEFINE_C_API_STRUCT(AiirPDLValue, const void);
DEFINE_C_API_STRUCT(AiirPDLResultList, void);

AIIR_CAPI_EXPORTED AiirPDLPatternModule
aiirPDLPatternModuleFromModule(AiirModule op);

AIIR_CAPI_EXPORTED void aiirPDLPatternModuleDestroy(AiirPDLPatternModule op);

AIIR_CAPI_EXPORTED AiirRewritePatternSet
aiirRewritePatternSetFromPDLPatternModule(AiirPDLPatternModule op);

/// Cast the AiirPDLValue to an AiirValue.
/// Return a null value if the cast fails, just like llvm::dyn_cast.
AIIR_CAPI_EXPORTED AiirValue aiirPDLValueAsValue(AiirPDLValue value);

/// Cast the AiirPDLValue to an AiirType.
/// Return a null value if the cast fails, just like llvm::dyn_cast.
AIIR_CAPI_EXPORTED AiirType aiirPDLValueAsType(AiirPDLValue value);

/// Cast the AiirPDLValue to an AiirOperation.
/// Return a null value if the cast fails, just like llvm::dyn_cast.
AIIR_CAPI_EXPORTED AiirOperation aiirPDLValueAsOperation(AiirPDLValue value);

/// Cast the AiirPDLValue to an AiirAttribute.
/// Return a null value if the cast fails, just like llvm::dyn_cast.
AIIR_CAPI_EXPORTED AiirAttribute aiirPDLValueAsAttribute(AiirPDLValue value);

/// Push the AiirValue into the given AiirPDLResultList.
AIIR_CAPI_EXPORTED void
aiirPDLResultListPushBackValue(AiirPDLResultList results, AiirValue value);

/// Push the AiirType into the given AiirPDLResultList.
AIIR_CAPI_EXPORTED void aiirPDLResultListPushBackType(AiirPDLResultList results,
                                                      AiirType value);

/// Push the AiirOperation into the given AiirPDLResultList.
AIIR_CAPI_EXPORTED void
aiirPDLResultListPushBackOperation(AiirPDLResultList results,
                                   AiirOperation value);

/// Push the AiirAttribute into the given AiirPDLResultList.
AIIR_CAPI_EXPORTED void
aiirPDLResultListPushBackAttribute(AiirPDLResultList results,
                                   AiirAttribute value);

/// This function type is used as callbacks for PDL native rewrite functions.
/// Input values can be accessed by `values` with its size `nValues`;
/// output values can be added into `results` by `aiirPDLResultListPushBack*`
/// APIs. And the return value indicates whether the rewrite succeeds.
typedef AiirLogicalResult (*AiirPDLRewriteFunction)(
    AiirPatternRewriter rewriter, AiirPDLResultList results, size_t nValues,
    AiirPDLValue *values, void *userData);

/// Register a rewrite function into the given PDL pattern module.
/// `userData` will be provided as an argument to the rewrite function.
AIIR_CAPI_EXPORTED void aiirPDLPatternModuleRegisterRewriteFunction(
    AiirPDLPatternModule pdlModule, AiirStringRef name,
    AiirPDLRewriteFunction rewriteFn, void *userData);

/// This function type is used as callbacks for PDL native constraint functions.
/// Input values can be accessed by `values` with its size `nValues`;
/// output values can be added into `results` by `aiirPDLResultListPushBack*`
/// APIs. And the return value indicates whether the constraint holds.
typedef AiirLogicalResult (*AiirPDLConstraintFunction)(
    AiirPatternRewriter rewriter, AiirPDLResultList results, size_t nValues,
    AiirPDLValue *values, void *userData);

/// Register a constraint function into the given PDL pattern module.
/// `userData` will be provided as an argument to the constraint function.
AIIR_CAPI_EXPORTED void aiirPDLPatternModuleRegisterConstraintFunction(
    AiirPDLPatternModule pdlModule, AiirStringRef name,
    AiirPDLConstraintFunction constraintFn, void *userData);

#endif // AIIR_ENABLE_PDL_IN_PATTERNMATCH

#undef DEFINE_C_API_STRUCT

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_REWRITE_H
