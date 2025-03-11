//===-- LowerRepackArrays.cpp
//------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/CodeGen.h"

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Allocatable.h"
#include "flang/Optimizer/Builder/Runtime/Transformational.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_LOWERREPACKARRAYSPASS
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"
} // namespace fir

#define DEBUG_TYPE "lower-repack-arrays"

namespace {
class RepackArrayConversion {
public:
  RepackArrayConversion(std::optional<mlir::DataLayout> dataLayout)
      : dataLayout(dataLayout) {}

protected:
  std::optional<mlir::DataLayout> dataLayout;

  static bool canAllocateTempOnStack(mlir::Value box);
};

class PackArrayConversion : public mlir::OpRewritePattern<fir::PackArrayOp>,
                            RepackArrayConversion {
public:
  using OpRewritePattern::OpRewritePattern;

  PackArrayConversion(mlir::MLIRContext *context,
                      std::optional<mlir::DataLayout> dataLayout)
      : OpRewritePattern(context), RepackArrayConversion(dataLayout) {}

  mlir::LogicalResult
  matchAndRewrite(fir::PackArrayOp op,
                  mlir::PatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef bufferName = ".repacked";

  static mlir::Value allocateTempBuffer(fir::FirOpBuilder &builder,
                                        mlir::Location loc, bool useStack,
                                        mlir::Value origBox,
                                        llvm::ArrayRef<mlir::Value> extents,
                                        llvm::ArrayRef<mlir::Value> typeParams);
};

class UnpackArrayConversion : public mlir::OpRewritePattern<fir::UnpackArrayOp>,
                              RepackArrayConversion {
public:
  using OpRewritePattern::OpRewritePattern;

  UnpackArrayConversion(mlir::MLIRContext *context,
                        std::optional<mlir::DataLayout> dataLayout)
      : OpRewritePattern(context), RepackArrayConversion(dataLayout) {}

  mlir::LogicalResult
  matchAndRewrite(fir::UnpackArrayOp op,
                  mlir::PatternRewriter &rewriter) const override;
};
} // anonymous namespace

bool RepackArrayConversion::canAllocateTempOnStack(mlir::Value box) {
  return !fir::isPolymorphicType(box.getType());
}

mlir::LogicalResult
PackArrayConversion::matchAndRewrite(fir::PackArrayOp op,
                                     mlir::PatternRewriter &rewriter) const {
  mlir::Location loc = op.getLoc();
  fir::FirOpBuilder builder(rewriter, op.getOperation());
  if (op.getMaxSize() || op.getMaxElementSize() || op.getMinStride())
    TODO(loc, "fir.pack_array with constraints");
  if (op.getHeuristics() != fir::PackArrayHeuristics::None)
    TODO(loc, "fir.pack_array with heuristics");

  mlir::Value box = op.getArray();
  llvm::SmallVector<mlir::Value> typeParams(op.getTypeparams().begin(),
                                            op.getTypeparams().end());
  // TODO: set non-default lower bounds on fir.pack_array,
  // so that we can preserve lower bounds in the temporary box.
  fir::BoxValue boxValue(box, /*lbounds=*/{}, typeParams);
  mlir::Type boxType = boxValue.getBoxTy();
  unsigned rank = boxValue.rank();
  mlir::Type indexType = builder.getIndexType();
  mlir::Value zero = fir::factory::createZeroValue(builder, loc, indexType);

  // Fetch the extents from the box, and see if the array
  // is not empty.
  // If the type params are not explicitly provided, then we must also
  // fetch the type parameters from the box.
  //
  // bool isNotEmpty;
  // vector<int64_t> extents;
  // if (IsPresent(box) && !IsContiguous[UpTo](box[, 1])) {
  //   isNotEmpty = box->base_addr != null;
  //   extents = SHAPE(box);
  // } else {
  //   isNotEmpty = false;
  //   extents = vector<int64_t>(rank, 0);
  // }

  unsigned numTypeParams = 0;
  if (typeParams.size() == 0) {
    if (auto recordType = mlir::dyn_cast<fir::RecordType>(boxValue.getEleTy()))
      if (recordType.getNumLenParams() != 0)
        TODO(loc,
             "allocating temporary for a parameterized derived type array");

    if (auto charType = mlir::dyn_cast<fir::CharacterType>(boxValue.getEleTy()))
      if (charType.hasDynamicLen())
        numTypeParams = 1;
  }

  // For now we have to always check if the box is present.
  mlir::Type predicateType = builder.getI1Type();
  auto isPresent =
      builder.create<fir::IsPresentOp>(loc, predicateType, boxValue.getAddr());

  // The results of the IfOp are:
  //   (extent1, ..., extentN, typeParam1, ..., typeParamM, isNotEmpty)
  // The number of results is rank + numTypeParams + 1.
  llvm::SmallVector<mlir::Type> ifTypes(rank + numTypeParams, indexType);
  ifTypes.push_back(predicateType);
  llvm::SmallVector<mlir::Value> negativeResult(rank + numTypeParams, zero);
  negativeResult.push_back(
      fir::factory::createZeroValue(builder, loc, predicateType));
  bool failedTypeParams = false;
  llvm::SmallVector<mlir::Value> extentsAndPredicate =
      builder
          .genIfOp(loc, ifTypes, isPresent,
                   /*withElseRegion=*/true)
          .genThen([&]() {
            // The box is present.
            auto isContiguous = builder.create<fir::IsContiguousBoxOp>(
                loc, box, op.getInnermost());
            llvm::SmallVector<mlir::Value> extentsAndPredicate =
                builder
                    .genIfOp(loc, ifTypes, isContiguous,
                             /*withElseRegion=*/true)
                    .genThen([&]() {
                      // Box is contiguous, return zero.
                      builder.create<fir::ResultOp>(loc, negativeResult);
                    })
                    .genElse([&]() {
                      // Get the extents.
                      llvm::SmallVector<mlir::Value> results =
                          fir::factory::readExtents(builder, loc, boxValue);

                      // Get the type parameters from the box, if needed.
                      llvm::SmallVector<mlir::Value> assumedTypeParams;
                      if (numTypeParams != 0) {
                        if (auto charType = mlir::dyn_cast<fir::CharacterType>(
                                boxValue.getEleTy()))
                          if (charType.hasDynamicLen()) {
                            fir::factory::CharacterExprHelper charHelper(
                                builder, loc);
                            mlir::Value len = charHelper.readLengthFromBox(
                                boxValue.getAddr(), charType);
                            assumedTypeParams.push_back(
                                builder.createConvert(loc, indexType, len));
                          }

                        if (numTypeParams != assumedTypeParams.size()) {
                          failedTypeParams = true;
                          assumedTypeParams.append(
                              numTypeParams - assumedTypeParams.size(), zero);
                        }
                      }
                      results.append(assumedTypeParams);

                      auto dataAddr = builder.create<fir::BoxAddrOp>(
                          loc, boxValue.getMemTy(), boxValue.getAddr());
                      auto isNotEmpty = builder.create<fir::IsPresentOp>(
                          loc, predicateType, dataAddr);
                      results.push_back(isNotEmpty);
                      builder.create<fir::ResultOp>(loc, results);
                    })
                    .getResults();

            builder.create<fir::ResultOp>(loc, extentsAndPredicate);
          })
          .genElse([&]() {
            // Box is absent, nothing to do.
            builder.create<fir::ResultOp>(loc, negativeResult);
          })
          .getResults();

  if (failedTypeParams)
    return emitError(loc) << "failed to compute the type parameters for "
                          << op.getOperation() << '\n';

  // The last result is the isNotEmpty predicate value.
  mlir::Value isNotEmpty = extentsAndPredicate.pop_back_val();
  // If fir.pack_array does not specify type parameters, but they are needed
  // for the type, then use the parameters fetched from the box.
  if (typeParams.size() == 0 && numTypeParams != 0) {
    assert(extentsAndPredicate.size() > numTypeParams);
    typeParams.append(extentsAndPredicate.end() - numTypeParams,
                      extentsAndPredicate.end());
    extentsAndPredicate.pop_back_n(numTypeParams);
  }
  // The remaining resulst are the extents.
  llvm::SmallVector<mlir::Value> extents = std::move(extentsAndPredicate);
  assert(extents.size() == rank);

  mlir::Value tempBox;
  // Allocate memory for the temporary, if allocating on stack.
  // We can do it unconditionally, even if size is zero.
  if (op.getStack() && canAllocateTempOnStack(boxValue.getAddr())) {
    tempBox = allocateTempBuffer(builder, loc, /*useStack=*/true,
                                 boxValue.getAddr(), extents, typeParams);
    if (!tempBox)
      return rewriter.notifyMatchFailure(op,
                                         "failed to produce stack allocation");
  }

  mlir::Value newResult =
      builder.genIfOp(loc, {boxType}, isNotEmpty, /*withElseRegion=*/true)
          .genThen([&]() {
            // Do the heap allocation conditionally.
            if (!tempBox)
              tempBox =
                  allocateTempBuffer(builder, loc, /*useStack=*/false,
                                     boxValue.getAddr(), extents, typeParams);

            // Do the copy, if needed, and return the new box (shaped same way
            // as the original one).
            if (!op.getNoCopy())
              fir::runtime::genShallowCopy(builder, loc, tempBox,
                                           boxValue.getAddr(),
                                           /*resultIsAllocated=*/true);

            // Set the lower bounds after the original box.
            mlir::Value shape;
            if (!boxValue.getLBounds().empty()) {
              shape = builder.genShape(loc, boxValue.getLBounds(), extents);
            }

            // Rebox the temporary box to make its type the same as
            // the original box's.
            tempBox = builder.create<fir::ReboxOp>(loc, boxType, tempBox, shape,
                                                   /*slice=*/nullptr);
            builder.create<fir::ResultOp>(loc, tempBox);
          })
          .genElse([&]() {
            // Return original box.
            builder.create<fir::ResultOp>(loc, boxValue.getAddr());
          })
          .getResults()[0];

  rewriter.replaceOp(op, newResult);
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

  if (fir::isPolymorphicType(origBox.getType())) {
    // Use runtime to allocate polymorphic temporary vector using the dynamic
    // type of the original box and the provided numElements.
    // TODO: try to generalize it with BufferizeHLFIR.cpp:createArrayTemp().

    // We cannot allocate polymorphic entity on stack.
    // Return null, and allow the caller to reissue the call.
    if (useStack)
      return nullptr;

    mlir::Type indexType = builder.getIndexType();
    mlir::Type boxHeapType = fir::HeapType::get(tempType);
    mlir::Value boxAlloc = fir::factory::genNullBoxStorage(
        builder, loc, fir::ClassType::get(boxHeapType));
    fir::runtime::genAllocatableApplyMold(builder, loc, boxAlloc, origBox,
                                          tempType.getDimension());
    mlir::Value one = builder.createIntegerConstant(loc, indexType, 1);
    unsigned dim = 0;
    for (mlir::Value extent : extents) {
      mlir::Value dimIndex =
          builder.createIntegerConstant(loc, indexType, dim++);
      fir::runtime::genAllocatableSetBounds(builder, loc, boxAlloc, dimIndex,
                                            one, extent);
    }

    if (!typeParams.empty()) {
      // We should call AllocatableSetDerivedLength() here.
      TODO(loc,
           "polymorphic type with length parameters in PackArrayConversion");
    }

    fir::runtime::genAllocatableAllocate(builder, loc, boxAlloc);
    return builder.create<fir::LoadOp>(loc, boxAlloc);
  }

  // Allocate non-polymorphic temporary on stack or in heap.
  mlir::Value newBuffer;
  if (useStack)
    newBuffer =
        builder.createTemporary(loc, tempType, bufferName, extents, typeParams);
  else
    newBuffer = builder.createHeapTemporary(loc, tempType, bufferName, extents,
                                            typeParams);

  mlir::Type ptrType = newBuffer.getType();
  mlir::Type tempBoxType = fir::BoxType::get(mlir::isa<fir::HeapType>(ptrType)
                                                 ? ptrType
                                                 : fir::unwrapRefType(ptrType));
  mlir::Value shape = builder.genShape(loc, extents);
  mlir::Value newBox =
      builder.createBox(loc, tempBoxType, newBuffer, shape, /*slice=*/nullptr,
                        typeParams, /*tdesc=*/nullptr);
  return newBox;
}

mlir::LogicalResult
UnpackArrayConversion::matchAndRewrite(fir::UnpackArrayOp op,
                                       mlir::PatternRewriter &rewriter) const {
  mlir::Location loc = op.getLoc();
  fir::FirOpBuilder builder(rewriter, op.getOperation());
  mlir::Type predicateType = builder.getI1Type();
  mlir::Type indexType = builder.getIndexType();
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
    mlir::Value tempAddrAsIndex =
        builder.createConvert(loc, indexType, tempAddr);
    mlir::Value originalAddr =
        builder.create<fir::BoxAddrOp>(loc, addrType, originalBox);
    originalAddr = builder.createConvert(loc, indexType, originalAddr);

    auto isNotSame = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, tempAddrAsIndex, originalAddr);
    builder.genIfThen(loc, isNotSame).genThen([&]() {});
    // Copy from temporary to the original.
    if (!op.getNoCopy())
      fir::runtime::genShallowCopy(builder, loc, originalBox, tempBox,
                                   /*resultIsAllocated=*/true);

    // Deallocate, if it was allocated in heap.
    if (!op.getStack())
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
    std::optional<mlir::DataLayout> dl = fir::support::getOrSetMLIRDataLayout(
        module, /*allowDefaultLayout=*/false);
    mlir::RewritePatternSet patterns(context);
    patterns.insert<PackArrayConversion>(context, dl);
    patterns.insert<UnpackArrayConversion>(context, dl);
    mlir::GreedyRewriteConfig config;
    config.enableRegionSimplification =
        mlir::GreedySimplifyRegionLevel::Disabled;
    (void)applyPatternsGreedily(module, std::move(patterns), config);
  }
};

} // anonymous namespace
