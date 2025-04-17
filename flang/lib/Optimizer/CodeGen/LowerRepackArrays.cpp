//===-- LowerRepackArrays.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This pass expands fir.pack_array and fir.unpack_array operations
/// into sequences of other FIR operations and Fortran runtime calls.
/// This pass is using structured control flow FIR operations such
/// as fir.if, so its placement in the pipeline should guarantee
/// further lowering of these operations.
///
/// A fir.pack_array operation is converted into a sequence of checks
/// identifying whether an array needs to be copied into a contiguous
/// temporary. When the checks pass, a new memory allocation is done
/// for the temporary array (in either stack or heap memory).
/// If `fir.pack_array` does not have no_copy attribute, then
/// the original array is shallow-copied into the temporary.
///
/// A fir.unpack_array operations is converted into a check
/// of whether the original and the temporary arrays are different
/// memory. When the check passes, the temporary array might be
/// shallow-copied into the original array, and then the temporary
/// array is deallocated (if it was allocated in stack memory,
/// then there is no explicit deallocation).
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/CodeGen.h"

#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Allocatable.h"
#include "flang/Optimizer/Builder/Runtime/Transformational.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/OpenACC/RegisterOpenACCExtensions.h"
#include "flang/Optimizer/OpenMP/Support/RegisterOpenMPExtensions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_LOWERREPACKARRAYSPASS
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"
} // namespace fir

#define DEBUG_TYPE "lower-repack-arrays"

namespace {
class PackArrayConversion : public mlir::OpRewritePattern<fir::PackArrayOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::PackArrayOp op,
                  mlir::PatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef bufferName = ".repacked";

  // Return value of fir::BaseBoxType that represents a temporary
  // array created for the original box with given extents and
  // type parameters. The new box has the default lower bounds.
  // If useStack is true, then the temporary will be allocated
  // in stack memory (when possible).
  static mlir::Value allocateTempBuffer(fir::FirOpBuilder &builder,
                                        mlir::Location loc, bool useStack,
                                        mlir::Value origBox,
                                        llvm::ArrayRef<mlir::Value> extents,
                                        llvm::ArrayRef<mlir::Value> typeParams);

  // Generate value of fir::BaseBoxType that represents the result
  // of the given fir.pack_array operation. The original box
  // is assumed to be present (though, it may represent an empty array).
  static mlir::FailureOr<mlir::Value> genRepackedBox(fir::FirOpBuilder &builder,
                                                     mlir::Location loc,
                                                     fir::PackArrayOp packOp);
};

class UnpackArrayConversion
    : public mlir::OpRewritePattern<fir::UnpackArrayOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::UnpackArrayOp op,
                  mlir::PatternRewriter &rewriter) const override;
};
} // anonymous namespace

// Return true iff for the given original boxed array we can
// allocate temporary memory in stack memory.
// This function is used to synchronize allocation/deallocation
// implied by fir.pack_array and fir.unpack_array, because
// the presence of the stack attribute does not automatically
// mean that the allocation is actually done in stack memory.
// For example, we always do the heap allocation for polymorphic
// types using Fortran runtime.
// Adding the polymorpic mold to fir.alloca and then using
// Fortran runtime to compute the allocation size could probably
// resolve this limitation.
static bool canAllocateTempOnStack(mlir::Value box) {
  return !fir::isPolymorphicType(box.getType());
}

/// Return true if array repacking is safe either statically
/// (there are no 'is_safe' attributes) or dynamically
/// (neither of the 'is_safe' attributes claims 'isDynamicallySafe() == false').
/// \p op is either fir.pack_array or fir.unpack_array.
template <typename OP>
static bool repackIsSafe(OP op) {
  bool isSafe = true;
  if (auto isSafeAttrs = op.getIsSafe()) {
    // We currently support only the attributes for which
    // isDynamicallySafe() returns false.
    for (auto attr : *isSafeAttrs) {
      auto iface = mlir::cast<fir::SafeTempArrayCopyAttrInterface>(attr);
      if (iface.isDynamicallySafe())
        TODO(op.getLoc(), "dynamically safe array repacking");
      else
        isSafe = false;
    }
  }
  return isSafe;
}

mlir::LogicalResult
PackArrayConversion::matchAndRewrite(fir::PackArrayOp op,
                                     mlir::PatternRewriter &rewriter) const {
  mlir::Value box = op.getArray();
  // If repacking is not safe, then just use the original box.
  if (!repackIsSafe(op)) {
    rewriter.replaceOp(op, box);
    return mlir::success();
  }

  mlir::Location loc = op.getLoc();
  fir::FirOpBuilder builder(rewriter, op.getOperation());
  if (op.getMaxSize() || op.getMaxElementSize() || op.getMinStride())
    TODO(loc, "fir.pack_array with constraints");
  if (op.getHeuristics() != fir::PackArrayHeuristics::None)
    TODO(loc, "fir.pack_array with heuristics");

  auto boxType = mlir::cast<fir::BaseBoxType>(box.getType());

  // For now we have to always check if the box is present.
  auto isPresent =
      builder.create<fir::IsPresentOp>(loc, builder.getI1Type(), box);

  fir::IfOp ifOp = builder.create<fir::IfOp>(loc, boxType, isPresent,
                                             /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  // The box is present.
  auto newBox = genRepackedBox(builder, loc, op);
  if (mlir::failed(newBox))
    return newBox;
  builder.create<fir::ResultOp>(loc, *newBox);

  // The box is not present. Return original box.
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  builder.create<fir::ResultOp>(loc, box);

  rewriter.replaceOp(op, ifOp.getResult(0));
  return mlir::success();
}

mlir::Value PackArrayConversion::allocateTempBuffer(
    fir::FirOpBuilder &builder, mlir::Location loc, bool useStack,
    mlir::Value origBox, llvm::ArrayRef<mlir::Value> extents,
    llvm::ArrayRef<mlir::Value> typeParams) {
  auto tempType = mlir::cast<fir::SequenceType>(
      fir::extractSequenceType(origBox.getType()));
  assert(tempType.getDimension() == extents.size() &&
         "number of extents does not match the rank");

  mlir::Value shape = builder.genShape(loc, extents);
  auto [base, isHeapAllocation] = builder.createArrayTemp(
      loc, tempType, shape, extents, typeParams,
      fir::FirOpBuilder::genTempDeclareOp,
      fir::isPolymorphicType(origBox.getType()) ? origBox : nullptr, useStack,
      bufferName);
  // Make sure canAllocateTempOnStack() can recognize when
  // the temporary is actually allocated on the stack
  // by createArrayTemp(). Otherwise, we may miss dynamic
  // deallocation when lowering fir.unpack_array.
  if (useStack && canAllocateTempOnStack(origBox))
    assert(!isHeapAllocation && "temp must have been allocated on the stack");

  if (isHeapAllocation)
    if (auto baseType = mlir::dyn_cast<fir::ReferenceType>(base.getType()))
      if (mlir::isa<fir::BaseBoxType>(baseType.getEleTy()))
        return builder.create<fir::LoadOp>(loc, base);

  mlir::Type ptrType = base.getType();
  mlir::Type tempBoxType = fir::BoxType::get(mlir::isa<fir::HeapType>(ptrType)
                                                 ? ptrType
                                                 : fir::unwrapRefType(ptrType));
  mlir::Value newBox =
      builder.createBox(loc, tempBoxType, base, shape, /*slice=*/nullptr,
                        typeParams, /*tdesc=*/nullptr);
  return newBox;
}

mlir::FailureOr<mlir::Value>
PackArrayConversion::genRepackedBox(fir::FirOpBuilder &builder,
                                    mlir::Location loc, fir::PackArrayOp op) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Value box = op.getArray();

  llvm::SmallVector<mlir::Value> typeParams(op.getTypeparams().begin(),
                                            op.getTypeparams().end());
  auto boxType = mlir::cast<fir::BaseBoxType>(box.getType());
  mlir::Type indexType = builder.getIndexType();

  // If type parameters are not specified by fir.pack_array,
  // figure out how many of them we need to read from the box.
  unsigned numTypeParams = 0;
  if (typeParams.size() == 0) {
    if (auto recordType =
            mlir::dyn_cast<fir::RecordType>(boxType.unwrapInnerType()))
      if (recordType.getNumLenParams() != 0)
        TODO(loc,
             "allocating temporary for a parameterized derived type array");

    if (auto charType =
            mlir::dyn_cast<fir::CharacterType>(boxType.unwrapInnerType())) {
      if (charType.hasDynamicLen()) {
        // Read one length parameter from the box.
        numTypeParams = 1;
      } else {
        // Place the constant length into typeParams.
        mlir::Value length =
            builder.createIntegerConstant(loc, indexType, charType.getLen());
        typeParams.push_back(length);
      }
    }
  }

  // Create a temporay iff the original is not contigous and is not empty.
  auto isNotContiguous = builder.genNot(
      loc, builder.create<fir::IsContiguousBoxOp>(loc, box, op.getInnermost()));
  auto dataAddr =
      builder.create<fir::BoxAddrOp>(loc, fir::boxMemRefType(boxType), box);
  auto isNotEmpty =
      builder.create<fir::IsPresentOp>(loc, builder.getI1Type(), dataAddr);
  auto doPack =
      builder.create<mlir::arith::AndIOp>(loc, isNotContiguous, isNotEmpty);

  fir::IfOp ifOp =
      builder.create<fir::IfOp>(loc, boxType, doPack, /*withElseRegion=*/true);

  // Return original box.
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  builder.create<fir::ResultOp>(loc, box);

  // Create a new box.
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  // Get lower bounds and extents from the box.
  llvm::SmallVector<mlir::Value, Fortran::common::maxRank> lbounds, extents;
  fir::factory::genDimInfoFromBox(builder, loc, box, &lbounds, &extents,
                                  /*strides=*/nullptr);
  // Get the type parameters from the box, if needed.
  llvm::SmallVector<mlir::Value> assumedTypeParams;
  if (numTypeParams != 0) {
    if (auto charType =
            mlir::dyn_cast<fir::CharacterType>(boxType.unwrapInnerType()))
      if (charType.hasDynamicLen()) {
        fir::factory::CharacterExprHelper charHelper(builder, loc);
        mlir::Value len = charHelper.readLengthFromBox(box, charType);
        typeParams.push_back(builder.createConvert(loc, indexType, len));
      }

    if (numTypeParams != typeParams.size())
      return emitError(loc) << "failed to compute the type parameters for "
                            << op.getOperation() << '\n';
  }

  mlir::Value tempBox =
      allocateTempBuffer(builder, loc, op.getStack(), box, extents, typeParams);
  if (!op.getNoCopy())
    fir::runtime::genShallowCopy(builder, loc, tempBox, box,
                                 /*resultIsAllocated=*/true);

  // Set lower bounds after the original box.
  mlir::Value shift = builder.genShift(loc, lbounds);
  tempBox = builder.create<fir::ReboxOp>(loc, boxType, tempBox, shift,
                                         /*slice=*/nullptr);
  builder.create<fir::ResultOp>(loc, tempBox);

  return ifOp.getResult(0);
}

mlir::LogicalResult
UnpackArrayConversion::matchAndRewrite(fir::UnpackArrayOp op,
                                       mlir::PatternRewriter &rewriter) const {
  // If repacking is not safe, then just remove the operation.
  if (!repackIsSafe(op)) {
    rewriter.eraseOp(op);
    return mlir::success();
  }

  mlir::Location loc = op.getLoc();
  fir::FirOpBuilder builder(rewriter, op.getOperation());
  mlir::Type predicateType = builder.getI1Type();
  mlir::Value tempBox = op.getTemp();
  mlir::Value originalBox = op.getOriginal();

  // For now we have to always check if the box is present.
  auto isPresent =
      builder.create<fir::IsPresentOp>(loc, predicateType, originalBox);

  builder.genIfThen(loc, isPresent).genThen([&]() {
    mlir::Type addrType =
        fir::HeapType::get(fir::extractSequenceType(tempBox.getType()));
    mlir::Value tempAddr =
        builder.create<fir::BoxAddrOp>(loc, addrType, tempBox);
    mlir::Value originalAddr =
        builder.create<fir::BoxAddrOp>(loc, addrType, originalBox);

    auto isNotSame = builder.genPtrCompare(loc, mlir::arith::CmpIPredicate::ne,
                                           tempAddr, originalAddr);
    builder.genIfThen(loc, isNotSame).genThen([&]() {});
    // Copy from temporary to the original.
    if (!op.getNoCopy())
      fir::runtime::genShallowCopy(builder, loc, originalBox, tempBox,
                                   /*resultIsAllocated=*/true);

    // Deallocate, if it was allocated in heap.
    // Note that the stack attribute does not always mean
    // that the allocation was actually done in stack memory.
    // There are currently cases where we delegate the allocation
    // to the runtime that uses heap memory, even when the stack
    // attribute is set on fir.pack_array.
    if (!op.getStack() || !canAllocateTempOnStack(originalBox))
      builder.create<fir::FreeMemOp>(loc, tempAddr);
  });
  rewriter.eraseOp(op);
  return mlir::success();
}

namespace {
class LowerRepackArraysPass
    : public fir::impl::LowerRepackArraysPassBase<LowerRepackArraysPass> {
public:
  using LowerRepackArraysPassBase<
      LowerRepackArraysPass>::LowerRepackArraysPassBase;

  void runOnOperation() override final {
    auto *context = &getContext();
    mlir::ModuleOp module = getOperation();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<PackArrayConversion>(context);
    patterns.insert<UnpackArrayConversion>(context);
    mlir::GreedyRewriteConfig config;
    config.enableRegionSimplification =
        mlir::GreedySimplifyRegionLevel::Disabled;
    (void)applyPatternsGreedily(module, std::move(patterns), config);
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    fir::acc::registerTransformationalAttrsDependentDialects(registry);
    fir::omp::registerTransformationalAttrsDependentDialects(registry);
  }
};

} // anonymous namespace
