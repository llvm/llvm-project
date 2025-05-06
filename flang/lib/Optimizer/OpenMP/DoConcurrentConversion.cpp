//===- DoConcurrentConversion.cpp -- map `DO CONCURRENT` to OpenMP loops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/DirectivesCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "flang/Support/OpenMP-utils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

#include <memory>
#include <utility>

namespace flangomp {
#define GEN_PASS_DEF_DOCONCURRENTCONVERSIONPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "do-concurrent-conversion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace Fortran {
namespace lower {
namespace omp {
namespace internal {
mlir::Value mapTemporaryValue(fir::FirOpBuilder &builder,
                              mlir::omp::TargetOp targetOp, mlir::Value val,
                              llvm::StringRef name) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(val);
  auto copyVal = builder.createTemporary(val.getLoc(), val.getType());
  builder.createStoreWithConvert(copyVal.getLoc(), val, copyVal);

  llvm::SmallVector<mlir::Value> bounds;
  builder.setInsertionPoint(targetOp);
  mlir::Value mapOp = Fortran::common::openmp::createMapInfoOp(
      builder, copyVal.getLoc(), copyVal,
      /*varPtrPtr=*/mlir::Value{}, name.str(), bounds,
      /*members=*/llvm::SmallVector<mlir::Value>{},
      /*membersIndex=*/mlir::ArrayAttr{},
      static_cast<std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
          llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT),
      mlir::omp::VariableCaptureKind::ByCopy, copyVal.getType());

  mlir::Region &targetRegion = targetOp.getRegion();

  auto argIface = llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*targetOp);
  unsigned insertIndex =
      argIface.getMapBlockArgsStart() + argIface.numMapBlockArgs();
  targetOp.getMapVarsMutable().append(mlir::ValueRange{mapOp});
  mlir::Value clonedValArg =
      targetRegion.insertArgument(insertIndex, mapOp.getType(), mapOp.getLoc());

  mlir::Block *targetEntryBlock = &targetRegion.getBlocks().front();
  builder.setInsertionPointToStart(targetEntryBlock);
  auto loadOp =
      builder.create<fir::LoadOp>(clonedValArg.getLoc(), clonedValArg);
  return loadOp.getResult();
}

/// Check if cloning the bounds introduced any dependency on the outer region.
/// If so, then either clone them as well if they are MemoryEffectFree, or else
/// copy them to a new temporary and add them to the map and block_argument
/// lists and replace their uses with the new temporary.
///
/// TODO: similar to the above functions, this is copied from OpenMP lowering
/// (in this case, from `genBodyOfTargetOp`). Once we move to a common lib for
/// these utils this will move as well.
void cloneOrMapRegionOutsiders(fir::FirOpBuilder &builder,
                               mlir::omp::TargetOp targetOp) {
  mlir::Region &targetRegion = targetOp.getRegion();
  mlir::Block *targetEntryBlock = &targetRegion.getBlocks().front();
  llvm::SetVector<mlir::Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(targetRegion, valuesDefinedAbove);

  while (!valuesDefinedAbove.empty()) {
    for (mlir::Value val : valuesDefinedAbove) {
      mlir::Operation *valOp = val.getDefiningOp();
      assert(valOp != nullptr);
      if (mlir::isMemoryEffectFree(valOp)) {
        mlir::Operation *clonedOp = valOp->clone();
        targetEntryBlock->push_front(clonedOp);
        assert(clonedOp->getNumResults() == 1);
        val.replaceUsesWithIf(
            clonedOp->getResult(0), [targetEntryBlock](mlir::OpOperand &use) {
              return use.getOwner()->getBlock() == targetEntryBlock;
            });
      } else {
        mlir::Value mappedTemp = mapTemporaryValue(builder, targetOp, val,
                                                   /*name=*/llvm::StringRef{});
        val.replaceUsesWithIf(
            mappedTemp, [targetEntryBlock](mlir::OpOperand &use) {
              return use.getOwner()->getBlock() == targetEntryBlock;
            });
      }
    }
    valuesDefinedAbove.clear();
    mlir::getUsedValuesDefinedAbove(targetRegion, valuesDefinedAbove);
  }
}
} // namespace internal
} // namespace omp
} // namespace lower
} // namespace Fortran

namespace {
namespace looputils {
/// Stores info needed about the induction/iteration variable for each `do
/// concurrent` in a loop nest.
struct InductionVariableInfo {
  InductionVariableInfo(fir::DoConcurrentLoopOp loop,
                        mlir::Value inductionVar) {
    populateInfo(loop, inductionVar);
  }

  /// The operation allocating memory for iteration variable.
  mlir::Operation *iterVarMemDef;
  /// the operation(s) updating the iteration variable with the current
  /// iteration number.
  llvm::SmallVector<mlir::Operation *, 2> indVarUpdateOps;

private:
  /// For the \p doLoop parameter, find the following:
  ///
  /// 1. The operation that declares its iteration variable or allocates memory
  /// for it. For example, give the following loop:
  /// ```
  ///   ...
  ///   %i:2 = hlfir.declare %0 {uniq_name = "_QFEi"} : ...
  ///   ...
  ///   fir.do_concurrent.loop (%ind_var) = (%lb) to (%ub) step (%s) {
  ///     %ind_var_conv = fir.convert %ind_var : (index) -> i32
  ///     fir.store %ind_var_conv to %i#1 : !fir.ref<i32>
  ///     ...
  ///   }
  /// ```
  ///
  /// This function sets the `iterVarMemDef` member to the `hlfir.declare` op
  /// for `%i`.
  ///
  /// 2. The operation(s) that update the loop's iteration variable from its
  /// induction variable. For the above example, the `indVarUpdateOps` is
  /// populated with the first 2 ops in the loop's body.
  ///
  /// Note: The current implementation is dependent on how flang emits loop
  /// bodies; which is sufficient for the current simple test/use cases. If this
  /// proves to be insufficient, this should be made more generic.
  void populateInfo(fir::DoConcurrentLoopOp loop, mlir::Value inductionVar) {
    mlir::Value result = nullptr;

    // Checks if a StoreOp is updating the memref of the loop's iteration
    // variable.
    auto isStoringIV = [&](fir::StoreOp storeOp) {
      // Direct store into the IV memref.
      if (storeOp.getValue() == inductionVar) {
        indVarUpdateOps.push_back(storeOp);
        return true;
      }

      // Indirect store into the IV memref.
      if (auto convertOp = mlir::dyn_cast<fir::ConvertOp>(
              storeOp.getValue().getDefiningOp())) {
        if (convertOp.getOperand() == inductionVar) {
          indVarUpdateOps.push_back(convertOp);
          indVarUpdateOps.push_back(storeOp);
          return true;
        }
      }

      return false;
    };

    for (mlir::Operation &op : loop) {
      if (auto storeOp = mlir::dyn_cast<fir::StoreOp>(op))
        if (isStoringIV(storeOp)) {
          result = storeOp.getMemref();
          break;
        }
    }

    assert(result != nullptr && result.getDefiningOp() != nullptr);
    iterVarMemDef = result.getDefiningOp();
  }
};

using InductionVariableInfos = llvm::SmallVector<InductionVariableInfo>;

/// Collect the list of values used inside the loop but defined outside of it.
void collectLoopLiveIns(fir::DoConcurrentLoopOp loop,
                        llvm::SmallVectorImpl<mlir::Value> &liveIns) {
  llvm::SmallDenseSet<mlir::Value> seenValues;
  llvm::SmallDenseSet<mlir::Operation *> seenOps;

  for (auto [lb, ub, st] : llvm::zip_equal(
           loop.getLowerBound(), loop.getUpperBound(), loop.getStep())) {
    liveIns.push_back(lb);
    liveIns.push_back(ub);
    liveIns.push_back(st);
  }

  mlir::visitUsedValuesDefinedAbove(
      loop.getRegion(), [&](mlir::OpOperand *operand) {
        if (!seenValues.insert(operand->get()).second)
          return;

        mlir::Operation *definingOp = operand->get().getDefiningOp();
        // We want to collect ops corresponding to live-ins only once.
        if (definingOp && !seenOps.insert(definingOp).second)
          return;

        liveIns.push_back(operand->get());
      });
}

/// Collects values that are local to a loop: "loop-local values". A loop-local
/// value is one that is used exclusively inside the loop but allocated outside
/// of it. This usually corresponds to temporary values that are used inside the
/// loop body for initialzing other variables for example.
///
/// See `flang/test/Transforms/DoConcurrent/locally_destroyed_temp.f90` for an
/// example of why we need this.
///
/// \param [in] doLoop - the loop within which the function searches for values
/// used exclusively inside.
///
/// \param [out] locals - the list of loop-local values detected for \p doLoop.
void collectLoopLocalValues(fir::DoConcurrentLoopOp loop,
                            llvm::SetVector<mlir::Value> &locals) {
  loop.walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands()) {
      if (locals.contains(operand))
        continue;

      bool isLocal = true;

      if (!mlir::isa_and_present<fir::AllocaOp>(operand.getDefiningOp()))
        continue;

      // Values defined inside the loop are not interesting since they do not
      // need to be localized.
      if (loop->isAncestor(operand.getDefiningOp()))
        continue;

      for (auto *user : operand.getUsers()) {
        if (!loop->isAncestor(user)) {
          isLocal = false;
          break;
        }
      }

      if (isLocal)
        locals.insert(operand);
    }
  });
}

/// For a "loop-local" value \p local within a loop's scope, localizes that
/// value within the scope of the parallel region the loop maps to. Towards that
/// end, this function moves the allocation of \p local within \p allocRegion.
///
/// \param local - the value used exclusively within a loop's scope (see
/// collectLoopLocalValues).
///
/// \param allocRegion - the parallel region where \p local's allocation will be
/// privatized.
///
/// \param rewriter - builder used for updating \p allocRegion.
static void localizeLoopLocalValue(mlir::Value local, mlir::Region &allocRegion,
                                   mlir::ConversionPatternRewriter &rewriter) {
  rewriter.moveOpBefore(local.getDefiningOp(), &allocRegion.front().front());
}
} // namespace looputils

class DoConcurrentConversion
    : public mlir::OpConversionPattern<fir::DoConcurrentOp> {
public:
  using mlir::OpConversionPattern<fir::DoConcurrentOp>::OpConversionPattern;

  DoConcurrentConversion(
      mlir::MLIRContext *context, bool mapToDevice,
      llvm::DenseSet<fir::DoConcurrentOp> &concurrentLoopsToSkip)
      : OpConversionPattern(context), mapToDevice(mapToDevice),
        concurrentLoopsToSkip(concurrentLoopsToSkip) {}

  mlir::LogicalResult
  matchAndRewrite(fir::DoConcurrentOp doLoop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    looputils::InductionVariableInfos ivInfos;
    auto loop = mlir::cast<fir::DoConcurrentLoopOp>(
        doLoop.getRegion().back().getTerminator());

    auto indVars = loop.getLoopInductionVars();
    assert(indVars.has_value());

    for (mlir::Value indVar : *indVars)
      ivInfos.emplace_back(loop, indVar);

    llvm::SmallVector<mlir::Value> loopNestLiveIns;
    looputils::collectLoopLiveIns(loop, loopNestLiveIns);
    assert(!loopNestLiveIns.empty());

    llvm::SetVector<mlir::Value> locals;
    looputils::collectLoopLocalValues(loop, locals);

    // We do not want to map "loop-local" values to the device through
    // `omp.map.info` ops. Therefore, we remove them from the list of live-ins.
    loopNestLiveIns.erase(llvm::remove_if(loopNestLiveIns,
                                          [&](mlir::Value liveIn) {
                                            return locals.contains(liveIn);
                                          }),
                          loopNestLiveIns.end());

    mlir::omp::TargetOp targetOp;
    mlir::omp::LoopNestOperands loopNestClauseOps;

    mlir::IRMapping mapper;

    if (mapToDevice) {
      mlir::ModuleOp module = doLoop->getParentOfType<mlir::ModuleOp>();
      bool isTargetDevice =
          llvm::cast<mlir::omp::OffloadModuleInterface>(*module)
              .getIsTargetDevice();

      mlir::omp::TargetOperands targetClauseOps;
      genLoopNestClauseOps(doLoop.getLoc(), rewriter, loop, mapper,
                           loopNestClauseOps,
                           isTargetDevice ? nullptr : &targetClauseOps);

      LiveInShapeInfoMap liveInShapeInfoMap;
      fir::FirOpBuilder builder(
          rewriter,
          fir::getKindMapping(doLoop->getParentOfType<mlir::ModuleOp>()));

      for (mlir::Value liveIn : loopNestLiveIns) {
        targetClauseOps.mapVars.push_back(
            genMapInfoOpForLiveIn(builder, liveIn));
        liveInShapeInfoMap.insert(
            {liveIn, TargetDeclareShapeCreationInfo(liveIn)});
      }

      targetOp =
          genTargetOp(doLoop.getLoc(), rewriter, mapper, loopNestLiveIns,
                      targetClauseOps, loopNestClauseOps, liveInShapeInfoMap);
      genTeamsOp(doLoop.getLoc(), rewriter);
    }

    mlir::omp::ParallelOp parallelOp =
        genParallelOp(doLoop.getLoc(), rewriter, ivInfos, mapper);
    // Only set as composite when part of `distribute parallel do`.
    parallelOp.setComposite(mapToDevice);

    if (!mapToDevice)
      genLoopNestClauseOps(doLoop.getLoc(), rewriter, loop, mapper,
                           loopNestClauseOps);

    for (mlir::Value local : locals)
      looputils::localizeLoopLocalValue(local, parallelOp.getRegion(),
                                        rewriter);

    if (mapToDevice)
      genDistributeOp(doLoop.getLoc(), rewriter).setComposite(/*val=*/true);

    mlir::omp::LoopNestOp ompLoopNest =
        genWsLoopOp(rewriter, loop, mapper, loopNestClauseOps,
                    /*isComposite=*/mapToDevice);

    rewriter.setInsertionPoint(doLoop);
    fir::FirOpBuilder builder(
        rewriter,
        fir::getKindMapping(doLoop->getParentOfType<mlir::ModuleOp>()));

    // Collect iteration variable(s) allocations so that we can move them
    // outside the `fir.do_concurrent` wrapper (before erasing it).
    llvm::SmallVector<mlir::Operation *> opsToMove;
    for (mlir::Operation &op : llvm::drop_end(doLoop))
      opsToMove.push_back(&op);

    mlir::Block *allocBlock = builder.getAllocaBlock();

    for (mlir::Operation *op : llvm::reverse(opsToMove)) {
      rewriter.moveOpBefore(op, allocBlock, allocBlock->begin());
    }

    // Mark `unordered` loops that are not perfectly nested to be skipped from
    // the legality check of the `ConversionTarget` since we are not interested
    // in mapping them to OpenMP.
    ompLoopNest->walk([&](fir::DoConcurrentOp doLoop) {
      concurrentLoopsToSkip.insert(doLoop);
    });

    rewriter.eraseOp(doLoop);

    return mlir::success();
  }

private:
  struct TargetDeclareShapeCreationInfo {
    // Note: We use `std::vector` (rather than `llvm::SmallVector` as usual) to
    // interface more easily `ShapeShiftOp::getOrigins()` which returns
    // `std::vector`.
    std::vector<mlir::Value> startIndices{};
    std::vector<mlir::Value> extents{};

    TargetDeclareShapeCreationInfo(mlir::Value liveIn) {
      mlir::Value shape = nullptr;
      mlir::Operation *liveInDefiningOp = liveIn.getDefiningOp();
      auto declareOp =
          mlir::dyn_cast_if_present<hlfir::DeclareOp>(liveInDefiningOp);

      if (declareOp != nullptr)
        shape = declareOp.getShape();

      if (shape == nullptr)
        return;

      auto shapeOp =
          mlir::dyn_cast_if_present<fir::ShapeOp>(shape.getDefiningOp());
      auto shapeShiftOp =
          mlir::dyn_cast_if_present<fir::ShapeShiftOp>(shape.getDefiningOp());

      if (shapeOp == nullptr && shapeShiftOp == nullptr)
        TODO(liveIn.getLoc(),
             "Shapes not defined by `fir.shape` or `fir.shape_shift` op's are"
             "not supported yet.");

      if (shapeShiftOp != nullptr)
        startIndices = shapeShiftOp.getOrigins();

      extents = shapeOp != nullptr
                    ? std::vector<mlir::Value>(shapeOp.getExtents().begin(),
                                               shapeOp.getExtents().end())
                    : shapeShiftOp.getExtents();
    }

    bool isShapedValue() const { return !extents.empty(); }
    bool isShapeShiftedValue() const { return !startIndices.empty(); }
  };

  using LiveInShapeInfoMap =
      llvm::DenseMap<mlir::Value, TargetDeclareShapeCreationInfo>;

  mlir::omp::ParallelOp
  genParallelOp(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
                looputils::InductionVariableInfos &ivInfos,
                mlir::IRMapping &mapper) const {
    auto parallelOp = rewriter.create<mlir::omp::ParallelOp>(loc);
    rewriter.createBlock(&parallelOp.getRegion());
    rewriter.setInsertionPoint(rewriter.create<mlir::omp::TerminatorOp>(loc));

    genLoopNestIndVarAllocs(rewriter, ivInfos, mapper);
    return parallelOp;
  }

  void genLoopNestIndVarAllocs(mlir::ConversionPatternRewriter &rewriter,
                               looputils::InductionVariableInfos &ivInfos,
                               mlir::IRMapping &mapper) const {

    for (auto &indVarInfo : ivInfos)
      genInductionVariableAlloc(rewriter, indVarInfo.iterVarMemDef, mapper);
  }

  mlir::Operation *
  genInductionVariableAlloc(mlir::ConversionPatternRewriter &rewriter,
                            mlir::Operation *indVarMemDef,
                            mlir::IRMapping &mapper) const {
    assert(
        indVarMemDef != nullptr &&
        "Induction variable memdef is expected to have a defining operation.");

    llvm::SmallSetVector<mlir::Operation *, 2> indVarDeclareAndAlloc;
    for (auto operand : indVarMemDef->getOperands())
      indVarDeclareAndAlloc.insert(operand.getDefiningOp());
    indVarDeclareAndAlloc.insert(indVarMemDef);

    mlir::Operation *result;
    for (mlir::Operation *opToClone : indVarDeclareAndAlloc)
      result = rewriter.clone(*opToClone, mapper);

    return result;
  }

  void genLoopNestClauseOps(
      mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
      fir::DoConcurrentLoopOp loop, mlir::IRMapping &mapper,
      mlir::omp::LoopNestOperands &loopNestClauseOps,
      mlir::omp::TargetOperands *targetClauseOps = nullptr) const {
    assert(loopNestClauseOps.loopLowerBounds.empty() &&
           "Loop nest bounds were already emitted!");

    auto populateBounds = [](mlir::Value var,
                             llvm::SmallVectorImpl<mlir::Value> &bounds) {
      bounds.push_back(var.getDefiningOp()->getResult(0));
    };

    auto hostEvalCapture = [&](mlir::Value var,
                               llvm::SmallVectorImpl<mlir::Value> &bounds) {
      populateBounds(var, bounds);

      if (targetClauseOps)
        targetClauseOps->hostEvalVars.push_back(var);
    };

    for (auto [lb, ub, st] : llvm::zip_equal(
             loop.getLowerBound(), loop.getUpperBound(), loop.getStep())) {
      hostEvalCapture(lb, loopNestClauseOps.loopLowerBounds);
      hostEvalCapture(ub, loopNestClauseOps.loopUpperBounds);
      hostEvalCapture(st, loopNestClauseOps.loopSteps);
    }

    loopNestClauseOps.loopInclusive = rewriter.getUnitAttr();
  }

  mlir::omp::LoopNestOp
  genWsLoopOp(mlir::ConversionPatternRewriter &rewriter,
              fir::DoConcurrentLoopOp loop, mlir::IRMapping &mapper,
              const mlir::omp::LoopNestOperands &clauseOps,
              bool isComposite) const {

    auto wsloopOp = rewriter.create<mlir::omp::WsloopOp>(loop.getLoc());
    wsloopOp.setComposite(isComposite);
    rewriter.createBlock(&wsloopOp.getRegion());

    auto loopNestOp =
        rewriter.create<mlir::omp::LoopNestOp>(loop.getLoc(), clauseOps);

    // Clone the loop's body inside the loop nest construct using the
    // mapped values.
    rewriter.cloneRegionBefore(loop.getRegion(), loopNestOp.getRegion(),
                               loopNestOp.getRegion().begin(), mapper);

    rewriter.setInsertionPointToEnd(&loopNestOp.getRegion().back());
    rewriter.create<mlir::omp::YieldOp>(loop->getLoc());

    return loopNestOp;
  }

  void genBoundsOps(fir::FirOpBuilder &builder, mlir::Value liveIn,
                    mlir::Value rawAddr,
                    llvm::SmallVectorImpl<mlir::Value> &boundsOps) const {
    fir::ExtendedValue extVal =
        hlfir::translateToExtendedValue(rawAddr.getLoc(), builder,
                                        hlfir::Entity{liveIn},
                                        /*contiguousHint=*/
                                        true)
            .first;
    fir::factory::AddrAndBoundsInfo info = fir::factory::getDataOperandBaseAddr(
        builder, rawAddr, /*isOptional=*/false, rawAddr.getLoc());
    boundsOps = fir::factory::genImplicitBoundsOps<mlir::omp::MapBoundsOp,
                                                   mlir::omp::MapBoundsType>(
        builder, info, extVal,
        /*dataExvIsAssumedSize=*/false, rawAddr.getLoc());
  }

  mlir::omp::MapInfoOp genMapInfoOpForLiveIn(fir::FirOpBuilder &builder,
                                             mlir::Value liveIn) const {
    mlir::Value rawAddr = liveIn;
    llvm::StringRef name;

    mlir::Operation *liveInDefiningOp = liveIn.getDefiningOp();
    auto declareOp =
        mlir::dyn_cast_if_present<hlfir::DeclareOp>(liveInDefiningOp);

    if (declareOp != nullptr) {
      // Use the raw address to avoid unboxing `fir.box` values whenever
      // possible. Put differently, if we have access to the direct value memory
      // reference/address, we use it.
      rawAddr = declareOp.getOriginalBase();
      name = declareOp.getUniqName();
    }

    if (!llvm::isa<mlir::omp::PointerLikeType>(rawAddr.getType())) {
      builder.setInsertionPointAfter(liveInDefiningOp);
      auto copyVal = builder.createTemporary(liveIn.getLoc(), liveIn.getType());
      builder.createStoreWithConvert(copyVal.getLoc(), liveIn, copyVal);
      rawAddr = copyVal;
    }

    mlir::Type liveInType = liveIn.getType();
    mlir::Type eleType = liveInType;
    if (auto refType = mlir::dyn_cast<fir::ReferenceType>(liveInType))
      eleType = refType.getElementType();

    llvm::omp::OpenMPOffloadMappingFlags mapFlag =
        llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;
    mlir::omp::VariableCaptureKind captureKind =
        mlir::omp::VariableCaptureKind::ByRef;

    if (fir::isa_trivial(eleType) || fir::isa_char(eleType)) {
      captureKind = mlir::omp::VariableCaptureKind::ByCopy;
    } else if (!fir::isa_builtin_cptr_type(eleType)) {
      mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
      mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
    }

    llvm::SmallVector<mlir::Value> boundsOps;
    genBoundsOps(builder, liveIn, rawAddr, boundsOps);

    return Fortran::common::openmp::createMapInfoOp(
        builder, liveIn.getLoc(), rawAddr,
        /*varPtrPtr=*/{}, name.str(), boundsOps,
        /*members=*/{},
        /*membersIndex=*/mlir::ArrayAttr{},
        static_cast<
            std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
            mapFlag),
        captureKind, rawAddr.getType());
  }

  mlir::omp::TargetOp
  genTargetOp(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
              mlir::IRMapping &mapper, llvm::ArrayRef<mlir::Value> mappedVars,
              mlir::omp::TargetOperands &clauseOps,
              mlir::omp::LoopNestOperands &loopNestClauseOps,
              const LiveInShapeInfoMap &liveInShapeInfoMap) const {
    auto targetOp = rewriter.create<mlir::omp::TargetOp>(loc, clauseOps);
    auto argIface = llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*targetOp);

    mlir::Region &region = targetOp.getRegion();

    llvm::SmallVector<mlir::Type> regionArgTypes;
    llvm::SmallVector<mlir::Location> regionArgLocs;

    for (auto var : llvm::concat<const mlir::Value>(clauseOps.hostEvalVars,
                                                    clauseOps.mapVars)) {
      regionArgTypes.push_back(var.getType());
      regionArgLocs.push_back(var.getLoc());
    }

    rewriter.createBlock(&region, {}, regionArgTypes, regionArgLocs);
    fir::FirOpBuilder builder(
        rewriter,
        fir::getKindMapping(targetOp->getParentOfType<mlir::ModuleOp>()));

    // Within the loop, it possible that we discover other values that need to
    // mapped to the target region (the shape info values for arrays, for
    // example). Therefore, the map block args might be extended and resized.
    // Hence, we invoke `argIface.getMapBlockArgs()` every iteration to make
    // sure we access the proper vector of data.
    int idx = 0;
    for (auto [mapInfoOp, mappedVar] :
         llvm::zip_equal(clauseOps.mapVars, mappedVars)) {
      auto miOp = mlir::cast<mlir::omp::MapInfoOp>(mapInfoOp.getDefiningOp());
      hlfir::DeclareOp liveInDeclare =
          genLiveInDeclare(builder, targetOp, argIface.getMapBlockArgs()[idx],
                           miOp, liveInShapeInfoMap.at(mappedVar));
      ++idx;

      // TODO If `mappedVar.getDefiningOp()` is a `fir::BoxAddrOp`, we probably
      // need to "unpack" the box by getting the defining op of it's value.
      // However, we did not hit this case in reality yet so leaving it as a
      // todo for now.

      auto mapHostValueToDevice = [&](mlir::Value hostValue,
                                      mlir::Value deviceValue) {
        if (!llvm::isa<mlir::omp::PointerLikeType>(hostValue.getType()))
          mapper.map(hostValue,
                     builder.loadIfRef(hostValue.getLoc(), deviceValue));
        else
          mapper.map(hostValue, deviceValue);
      };

      mapHostValueToDevice(mappedVar, liveInDeclare.getOriginalBase());

      if (auto origDeclareOp = mlir::dyn_cast_if_present<hlfir::DeclareOp>(
              mappedVar.getDefiningOp()))
        mapHostValueToDevice(origDeclareOp.getBase(), liveInDeclare.getBase());
    }

    for (auto [arg, hostEval] : llvm::zip_equal(argIface.getHostEvalBlockArgs(),
                                                clauseOps.hostEvalVars))
      mapper.map(hostEval, arg);

    for (unsigned i = 0; i < loopNestClauseOps.loopLowerBounds.size(); ++i) {
      loopNestClauseOps.loopLowerBounds[i] =
          mapper.lookup(loopNestClauseOps.loopLowerBounds[i]);
      loopNestClauseOps.loopUpperBounds[i] =
          mapper.lookup(loopNestClauseOps.loopUpperBounds[i]);
      loopNestClauseOps.loopSteps[i] =
          mapper.lookup(loopNestClauseOps.loopSteps[i]);
    }

    Fortran::lower::omp::internal::cloneOrMapRegionOutsiders(builder, targetOp);
    rewriter.setInsertionPoint(
        rewriter.create<mlir::omp::TerminatorOp>(targetOp.getLoc()));

    return targetOp;
  }

  hlfir::DeclareOp genLiveInDeclare(
      fir::FirOpBuilder &builder, mlir::omp::TargetOp targetOp,
      mlir::Value liveInArg, mlir::omp::MapInfoOp liveInMapInfoOp,
      const TargetDeclareShapeCreationInfo &targetShapeCreationInfo) const {
    mlir::Type liveInType = liveInArg.getType();
    std::string liveInName = liveInMapInfoOp.getName().has_value()
                                 ? liveInMapInfoOp.getName().value().str()
                                 : std::string("");
    if (fir::isa_ref_type(liveInType))
      liveInType = fir::unwrapRefType(liveInType);

    mlir::Value shape = [&]() -> mlir::Value {
      if (!targetShapeCreationInfo.isShapedValue())
        return {};

      llvm::SmallVector<mlir::Value> extentOperands;
      llvm::SmallVector<mlir::Value> startIndexOperands;

      if (targetShapeCreationInfo.isShapeShiftedValue()) {
        llvm::SmallVector<mlir::Value> shapeShiftOperands;

        size_t shapeIdx = 0;
        for (auto [startIndex, extent] :
             llvm::zip_equal(targetShapeCreationInfo.startIndices,
                             targetShapeCreationInfo.extents)) {
          shapeShiftOperands.push_back(
              Fortran::lower::omp::internal::mapTemporaryValue(
                  builder, targetOp, startIndex,
                  liveInName + ".start_idx.dim" + std::to_string(shapeIdx)));
          shapeShiftOperands.push_back(
              Fortran::lower::omp::internal::mapTemporaryValue(
                  builder, targetOp, extent,
                  liveInName + ".extent.dim" + std::to_string(shapeIdx)));
          ++shapeIdx;
        }

        auto shapeShiftType = fir::ShapeShiftType::get(
            builder.getContext(), shapeShiftOperands.size() / 2);
        return builder.create<fir::ShapeShiftOp>(
            liveInArg.getLoc(), shapeShiftType, shapeShiftOperands);
      }

      llvm::SmallVector<mlir::Value> shapeOperands;
      size_t shapeIdx = 0;
      for (auto extent : targetShapeCreationInfo.extents) {
        shapeOperands.push_back(
            Fortran::lower::omp::internal::mapTemporaryValue(
                builder, targetOp, extent,
                liveInName + ".extent.dim" + std::to_string(shapeIdx)));
        ++shapeIdx;
      }

      return builder.create<fir::ShapeOp>(liveInArg.getLoc(), shapeOperands);
    }();

    return builder.create<hlfir::DeclareOp>(liveInArg.getLoc(), liveInArg,
                                            liveInName, shape);
  }

  mlir::omp::TeamsOp
  genTeamsOp(mlir::Location loc,
             mlir::ConversionPatternRewriter &rewriter) const {
    auto teamsOp = rewriter.create<mlir::omp::TeamsOp>(
        loc, /*clauses=*/mlir::omp::TeamsOperands{});

    rewriter.createBlock(&teamsOp.getRegion());
    rewriter.setInsertionPoint(rewriter.create<mlir::omp::TerminatorOp>(loc));

    return teamsOp;
  }

  mlir::omp::DistributeOp
  genDistributeOp(mlir::Location loc,
                  mlir::ConversionPatternRewriter &rewriter) const {
    auto distOp = rewriter.create<mlir::omp::DistributeOp>(
        loc, /*clauses=*/mlir::omp::DistributeOperands{});

    rewriter.createBlock(&distOp.getRegion());
    return distOp;
  }

  bool mapToDevice;
  llvm::DenseSet<fir::DoConcurrentOp> &concurrentLoopsToSkip;
};

class DoConcurrentConversionPass
    : public flangomp::impl::DoConcurrentConversionPassBase<
          DoConcurrentConversionPass> {
public:
  DoConcurrentConversionPass() = default;

  DoConcurrentConversionPass(
      const flangomp::DoConcurrentConversionPassOptions &options)
      : DoConcurrentConversionPassBase(options) {}

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    if (func.isDeclaration())
      return;

    mlir::MLIRContext *context = &getContext();

    if (mapTo != flangomp::DoConcurrentMappingKind::DCMK_Host &&
        mapTo != flangomp::DoConcurrentMappingKind::DCMK_Device) {
      mlir::emitWarning(mlir::UnknownLoc::get(context),
                        "DoConcurrentConversionPass: invalid `map-to` value. "
                        "Valid values are: `host` or `device`");
      return;
    }

    llvm::DenseSet<fir::DoConcurrentOp> concurrentLoopsToSkip;
    mlir::RewritePatternSet patterns(context);
    patterns.insert<DoConcurrentConversion>(
        context, mapTo == flangomp::DoConcurrentMappingKind::DCMK_Device,
        concurrentLoopsToSkip);
    mlir::ConversionTarget target(*context);
    target.addDynamicallyLegalOp<fir::DoConcurrentOp>(
        [&](fir::DoConcurrentOp op) {
          return concurrentLoopsToSkip.contains(op);
        });
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });

    if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                               std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
flangomp::createDoConcurrentConversionPass(bool mapToDevice) {
  DoConcurrentConversionPassOptions options;
  options.mapTo = mapToDevice ? flangomp::DoConcurrentMappingKind::DCMK_Device
                              : flangomp::DoConcurrentMappingKind::DCMK_Host;

  return std::make_unique<DoConcurrentConversionPass>(options);
}
