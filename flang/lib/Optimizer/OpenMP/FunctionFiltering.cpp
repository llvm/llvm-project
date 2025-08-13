//===- FunctionFiltering.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements transforms to filter out functions intended for the host
// when compiling for the device and vice versa.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace flangomp {
#define GEN_PASS_DEF_FUNCTIONFILTERINGPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

using namespace mlir;

/// Add an operation to one of the output sets to be later rewritten, based on
/// whether it is located within the given region.
template <typename OpTy>
static void collectRewriteImpl(OpTy op, Region &region,
                               llvm::SetVector<OpTy> &rewrites,
                               llvm::SetVector<Operation *> *parentRewrites) {
  if (rewrites.contains(op))
    return;

  if (!parentRewrites || region.isAncestor(op->getParentRegion()))
    rewrites.insert(op);
  else
    parentRewrites->insert(op.getOperation());
}

template <typename OpTy>
static void collectRewrite(OpTy op, Region &region,
                           llvm::SetVector<OpTy> &rewrites,
                           llvm::SetVector<Operation *> *parentRewrites) {
  collectRewriteImpl(op, region, rewrites, parentRewrites);
}

/// Add an \c omp.map.info operation and all its members recursively to one of
/// the output sets to be later rewritten, based on whether they are located
/// within the given region.
///
/// Dependencies across \c omp.map.info are maintained by ensuring dependencies
/// are added to the output sets before operations based on them.
template <>
void collectRewrite(omp::MapInfoOp mapOp, Region &region,
                    llvm::SetVector<omp::MapInfoOp> &rewrites,
                    llvm::SetVector<Operation *> *parentRewrites) {
  for (Value member : mapOp.getMembers())
    collectRewrite(cast<omp::MapInfoOp>(member.getDefiningOp()), region,
                   rewrites, parentRewrites);

  collectRewriteImpl(mapOp, region, rewrites, parentRewrites);
}

/// Add the given value to a sorted set if it should be replaced by a
/// placeholder when used as an operand that must remain for the device. The
/// used output set used will depend on whether the value is defined within the
/// given region.
///
/// Values that are block arguments of \c omp.target_data and \c func.func
/// operations are skipped, since they will still be available after all
/// rewrites are completed.
static void collectRewrite(Value value, Region &region,
                           llvm::SetVector<Value> &rewrites,
                           llvm::SetVector<Value> *parentRewrites) {
  if ((isa<BlockArgument>(value) &&
       isa<func::FuncOp, omp::TargetDataOp>(
           cast<BlockArgument>(value).getOwner()->getParentOp())) ||
      rewrites.contains(value))
    return;

  if (!parentRewrites) {
    rewrites.insert(value);
    return;
  }

  Region *definingRegion;
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    definingRegion = blockArg.getOwner()->getParent();
  else
    definingRegion = value.getDefiningOp()->getParentRegion();

  assert(definingRegion && "defining op/block must exist in a region");

  if (region.isAncestor(definingRegion))
    rewrites.insert(value);
  else
    parentRewrites->insert(value);
}

/// Add operations in \c childRewrites to one of the output sets based on
/// whether they are located within the given region.
template <typename OpTy>
static void
applyChildRewrites(Region &region,
                   const llvm::SetVector<Operation *> &childRewrites,
                   llvm::SetVector<OpTy> &rewrites,
                   llvm::SetVector<Operation *> *parentRewrites) {
  for (Operation *rewrite : childRewrites)
    if (auto op = dyn_cast<OpTy>(*rewrite))
      collectRewrite(op, region, rewrites, parentRewrites);
}

/// Add values in \c childRewrites to one of the output sets based on
/// whether they are defined within the given region.
static void applyChildRewrites(Region &region,
                               const llvm::SetVector<Value> &childRewrites,
                               llvm::SetVector<Value> &rewrites,
                               llvm::SetVector<Value> *parentRewrites) {
  for (Value value : childRewrites)
    collectRewrite(value, region, rewrites, parentRewrites);
}

namespace {
class FunctionFilteringPass
    : public flangomp::impl::FunctionFilteringPassBase<FunctionFilteringPass> {
public:
  FunctionFilteringPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OpBuilder opBuilder(context);
    auto op = dyn_cast<omp::OffloadModuleInterface>(getOperation());
    if (!op || !op.getIsTargetDevice())
      return;

    op->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
      // Do not filter functions with target regions inside, because they have
      // to be available for both host and device so that regular and reverse
      // offloading can be supported.
      bool hasTargetRegion =
          funcOp
              ->walk<WalkOrder::PreOrder>(
                  [&](omp::TargetOp) { return WalkResult::interrupt(); })
              .wasInterrupted();

      omp::DeclareTargetDeviceType declareType =
          omp::DeclareTargetDeviceType::host;
      auto declareTargetOp =
          dyn_cast<omp::DeclareTargetInterface>(funcOp.getOperation());
      if (declareTargetOp && declareTargetOp.isDeclareTarget())
        declareType = declareTargetOp.getDeclareTargetDeviceType();

      // Filtering a function here means deleting it if it doesn't contain a
      // target region. Else we explicitly set the omp.declare_target
      // attribute. The second stage of function filtering at the MLIR to LLVM
      // IR translation level will remove functions that contain the target
      // region from the generated llvm IR.
      if (declareType == omp::DeclareTargetDeviceType::host) {
        SymbolTable::UseRange funcUses = *funcOp.getSymbolUses(op);
        for (SymbolTable::SymbolUse use : funcUses) {
          Operation *callOp = use.getUser();
          if (auto internalFunc = mlir::dyn_cast<func::FuncOp>(callOp)) {
            // Do not delete internal procedures holding the symbol of their
            // Fortran host procedure as attribute.
            internalFunc->removeAttr(fir::getHostSymbolAttrName());
            // Set public visibility so that the function is not deleted by MLIR
            // because unused. Changing it is OK here because the function will
            // be deleted anyway in the second filtering phase.
            internalFunc.setVisibility(mlir::SymbolTable::Visibility::Public);
            continue;
          }
          // If the callOp has users then replace them with Undef values.
          if (!callOp->use_empty()) {
            SmallVector<Value> undefResults;
            for (Value res : callOp->getResults()) {
              opBuilder.setInsertionPoint(callOp);
              undefResults.emplace_back(
                  fir::UndefOp::create(opBuilder, res.getLoc(), res.getType()));
            }
            callOp->replaceAllUsesWith(undefResults);
          }
          // Remove the callOp
          callOp->erase();
        }
        if (!hasTargetRegion) {
          funcOp.erase();
          return WalkResult::skip();
        }

        if (failed(rewriteHostRegion(funcOp.getRegion()))) {
          funcOp.emitOpError() << "could not be rewritten for target device";
          return WalkResult::interrupt();
        }

        if (declareTargetOp)
          declareTargetOp.setDeclareTarget(
              declareType, omp::DeclareTargetCaptureClause::to,
              declareTargetOp.getDeclareTargetAutomap());
      }
      return WalkResult::advance();
    });
  }

private:
  /// Rewrite the given host device region belonging to a function that contains
  /// \c omp.target operations, to remove host-only operations that are not used
  /// by device codegen.
  ///
  /// It is based on the expected form of the MLIR module as produced by Flang
  /// lowering and it performs the following mutations:
  ///   - Replace all values returned by the function with \c fir.undefined.
  ///   - Operations taking map-like clauses (e.g. \c omp.target,
  ///     \c omp.target_data, etc) are moved to the end of the function. If they
  ///     are nested inside of any other operations, they are hoisted out of
  ///     them. If the region belongs to \c omp.target_data, these operations
  ///     are hoisted to its top level, rather than to the parent function.
  ///   - \c device, \c if and \c depend clauses are removed from these target
  ///     functions. Values initializing other clauses are either replaced by
  ///     placeholders as follows:
  ///     - Values defined by block arguments are replaced by placeholders only
  ///       if they are not attached to \c func.func or \c omp.target_data
  ///       operations. In that case, they are kept unmodified.
  ///     - \c arith.constant and \c fir.address_of are maintained.
  ///     - Other values are replaced by a combination of an \c fir.alloca for a
  ///       single bit and an \c fir.convert to the original type of the value.
  ///       This can be done because the code eventually generated for these
  ///       operations will be discarded, as they aren't runnable by the target
  ///       device.
  ///   - \c omp.map.info operations associated to these target regions are
  ///     preserved. These are moved above all \c omp.target and sorted to
  ///     satisfy dependencies among them.
  ///   - \c bounds arguments are removed from \c omp.map.info operations.
  ///   - \c var_ptr and \c var_ptr_ptr arguments of \c omp.map.info are
  ///     handled as follows:
  ///     - \c var_ptr_ptr is expected to be defined by a \c fir.box_offset
  ///       operation which is preserved. Otherwise, the pass will fail.
  ///     - \c var_ptr can be defined by an \c hlfir.declare which is also
  ///       preserved. Its \c memref argument is replaced by a placeholder or
  ///       maintained similarly to non-map clauses of target operations
  ///       described above. If it has \c shape or \c typeparams arguments, they
  ///       are replaced by applicable constants. \c dummy_scope arguments
  ///       are discarded.
  ///   - Every other operation not located inside of an \c omp.target is
  ///     removed.
  ///   - Whenever a value or operation that would otherwise be replaced with a
  ///     placeholder is defined outside of the region being rewritten, it is
  ///     added to the \c parentOpRewrites or \c parentValRewrites output
  ///     argument, to be later handled by the caller. This is only intended to
  ///     properly support nested \c omp.target_data and \c omp.target placed
  ///     inside of \c omp.target_data. When called for the main function, these
  ///     output arguments must not be set.
  LogicalResult
  rewriteHostRegion(Region &region,
                    llvm::SetVector<Operation *> *parentOpRewrites = nullptr,
                    llvm::SetVector<Value> *parentValRewrites = nullptr) {
    // Extract parent op information.
    auto [funcOp, targetDataOp] = [&region]() {
      Operation *parent = region.getParentOp();
      return std::make_tuple(dyn_cast<func::FuncOp>(parent),
                             dyn_cast<omp::TargetDataOp>(parent));
    }();
    assert((bool)funcOp != (bool)targetDataOp &&
           "region must be defined by either func.func or omp.target_data");
    assert((bool)parentOpRewrites == (bool)targetDataOp &&
           (bool)parentValRewrites == (bool)targetDataOp &&
           "parent rewrites must be passed iff rewriting omp.target_data");

    // Collect operations that have mapping information associated to them.
    llvm::SmallVector<
        std::variant<omp::TargetOp, omp::TargetDataOp, omp::TargetEnterDataOp,
                     omp::TargetExitDataOp, omp::TargetUpdateOp>>
        targetOps;

    // Sets to store pending rewrites marked by child omp.target_data ops.
    llvm::SetVector<Operation *> childOpRewrites;
    llvm::SetVector<Value> childValRewrites;
    WalkResult result = region.walk<WalkOrder::PreOrder>([&](Operation *op) {
      // Skip the inside of omp.target regions, since these contain device
      // code.
      if (auto targetOp = dyn_cast<omp::TargetOp>(op)) {
        targetOps.push_back(targetOp);
        return WalkResult::skip();
      }

      if (auto targetOp = dyn_cast<omp::TargetDataOp>(op)) {
        // Recursively rewrite omp.target_data regions as well.
        if (failed(rewriteHostRegion(targetOp.getRegion(), &childOpRewrites,
                                     &childValRewrites))) {
          targetOp.emitOpError() << "rewrite for target device failed";
          return WalkResult::interrupt();
        }

        targetOps.push_back(targetOp);
        return WalkResult::skip();
      }

      if (auto targetOp = dyn_cast<omp::TargetEnterDataOp>(op))
        targetOps.push_back(targetOp);
      else if (auto targetOp = dyn_cast<omp::TargetExitDataOp>(op))
        targetOps.push_back(targetOp);
      else if (auto targetOp = dyn_cast<omp::TargetUpdateOp>(op))
        targetOps.push_back(targetOp);

      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return failure();

    // Make a temporary clone of the parent operation with an empty region,
    // and update all references to entry block arguments to those of the new
    // region. Users will later either be moved to the new region or deleted
    // when the original region is replaced by the new.
    OpBuilder builder(&getContext());
    builder.setInsertionPointAfter(region.getParentOp());
    Operation *newOp = builder.cloneWithoutRegions(*region.getParentOp());
    Block &block = newOp->getRegion(0).emplaceBlock();

    llvm::SmallVector<Location> locs;
    locs.reserve(region.getNumArguments());
    llvm::transform(region.getArguments(), std::back_inserter(locs),
                    [](const BlockArgument &arg) { return arg.getLoc(); });
    block.addArguments(region.getArgumentTypes(), locs);

    for (auto [oldArg, newArg] :
         llvm::zip_equal(region.getArguments(), block.getArguments()))
      oldArg.replaceAllUsesWith(newArg);

    // Collect omp.map.info ops while satisfying interdependencies. This must
    // be updated whenever operands to operations contained in targetOps change.
    llvm::SetVector<Value> rewriteValues;
    llvm::SetVector<omp::MapInfoOp> mapInfos;
    for (auto targetOp : targetOps) {
      std::visit(
          [&](auto op) {
            // Variables unused by the device, present on all target ops.
            op.getDeviceMutable().clear();
            op.getIfExprMutable().clear();

            for (Value mapVar : op.getMapVars())
              collectRewrite(cast<omp::MapInfoOp>(mapVar.getDefiningOp()),
                             region, mapInfos, parentOpRewrites);

            if constexpr (!std::is_same_v<decltype(op), omp::TargetDataOp>) {
              // Variables unused by the device, present on all target ops
              // except for omp.target_data.
              op.getDependVarsMutable().clear();
              op.setDependKindsAttr(nullptr);
            }

            if constexpr (std::is_same_v<decltype(op), omp::TargetOp>) {
              assert(op.getHostEvalVars().empty() &&
                     "unexpected host_eval in target device module");
              // TODO: Clear some of these operands rather than rewriting them,
              // depending on whether they are needed by device codegen once
              // support for them is fully implemented.
              for (Value allocVar : op.getAllocateVars())
                collectRewrite(allocVar, region, rewriteValues,
                               parentValRewrites);
              for (Value allocVar : op.getAllocatorVars())
                collectRewrite(allocVar, region, rewriteValues,
                               parentValRewrites);
              for (Value inReduction : op.getInReductionVars())
                collectRewrite(inReduction, region, rewriteValues,
                               parentValRewrites);
              for (Value isDevPtr : op.getIsDevicePtrVars())
                collectRewrite(isDevPtr, region, rewriteValues,
                               parentValRewrites);
              for (Value mapVar : op.getHasDeviceAddrVars())
                collectRewrite(cast<omp::MapInfoOp>(mapVar.getDefiningOp()),
                               region, mapInfos, parentOpRewrites);
              for (Value privateVar : op.getPrivateVars())
                collectRewrite(privateVar, region, rewriteValues,
                               parentValRewrites);
              if (Value threadLimit = op.getThreadLimit())
                collectRewrite(threadLimit, region, rewriteValues,
                               parentValRewrites);
            } else if constexpr (std::is_same_v<decltype(op),
                                                omp::TargetDataOp>) {
              for (Value mapVar : op.getUseDeviceAddrVars())
                collectRewrite(cast<omp::MapInfoOp>(mapVar.getDefiningOp()),
                               region, mapInfos, parentOpRewrites);
              for (Value mapVar : op.getUseDevicePtrVars())
                collectRewrite(cast<omp::MapInfoOp>(mapVar.getDefiningOp()),
                               region, mapInfos, parentOpRewrites);
            }
          },
          targetOp);
    }

    applyChildRewrites(region, childOpRewrites, mapInfos, parentOpRewrites);

    // Move omp.map.info ops to the new block and collect dependencies.
    llvm::SetVector<hlfir::DeclareOp> declareOps;
    llvm::SetVector<fir::BoxOffsetOp> boxOffsets;
    for (omp::MapInfoOp mapOp : mapInfos) {
      if (auto declareOp = dyn_cast_if_present<hlfir::DeclareOp>(
              mapOp.getVarPtr().getDefiningOp()))
        collectRewrite(declareOp, region, declareOps, parentOpRewrites);
      else
        collectRewrite(mapOp.getVarPtr(), region, rewriteValues,
                       parentValRewrites);

      if (Value varPtrPtr = mapOp.getVarPtrPtr()) {
        if (auto boxOffset = llvm::dyn_cast_if_present<fir::BoxOffsetOp>(
                varPtrPtr.getDefiningOp()))
          collectRewrite(boxOffset, region, boxOffsets, parentOpRewrites);
        else
          return mapOp->emitOpError() << "var_ptr_ptr rewrite only supported "
                                         "if defined by fir.box_offset";
      }

      // Bounds are not used during target device codegen.
      mapOp.getBoundsMutable().clear();
      mapOp->moveBefore(&block, block.end());
    }

    applyChildRewrites(region, childOpRewrites, declareOps, parentOpRewrites);
    applyChildRewrites(region, childOpRewrites, boxOffsets, parentOpRewrites);

    // Create a temporary marker to simplify the op moving process below.
    builder.setInsertionPointToStart(&block);
    auto marker = builder.create<fir::UndefOp>(builder.getUnknownLoc(),
                                               builder.getNoneType());
    builder.setInsertionPoint(marker);

    // Handle dependencies of hlfir.declare ops.
    for (hlfir::DeclareOp declareOp : declareOps) {
      collectRewrite(declareOp.getMemref(), region, rewriteValues,
                     parentValRewrites);

      // Shape and typeparams aren't needed for target device codegen, but
      // removing them would break verifiers.
      Value zero;
      if (declareOp.getShape() || !declareOp.getTypeparams().empty())
        zero = builder.create<arith::ConstantOp>(declareOp.getLoc(),
                                                 builder.getI64IntegerAttr(0));

      if (auto shape = declareOp.getShape()) {
        // The pre-cg rewrite pass requires the shape to be defined by one of
        // fir.shape, fir.shapeshift or fir.shift, so we need to make sure it's
        // still defined by one of these after this pass.
        Operation *shapeOp = shape.getDefiningOp();
        llvm::SmallVector<Value> extents(shapeOp->getNumOperands(), zero);
        Value newShape =
            llvm::TypeSwitch<Operation *, Value>(shapeOp)
                .Case([&](fir::ShapeOp op) {
                  return builder.create<fir::ShapeOp>(op.getLoc(), extents);
                })
                .Case([&](fir::ShapeShiftOp op) {
                  auto type = fir::ShapeShiftType::get(op.getContext(),
                                                       extents.size() / 2);
                  return builder.create<fir::ShapeShiftOp>(op.getLoc(), type,
                                                           extents);
                })
                .Case([&](fir::ShiftOp op) {
                  auto type =
                      fir::ShiftType::get(op.getContext(), extents.size());
                  return builder.create<fir::ShiftOp>(op.getLoc(), type,
                                                      extents);
                })
                .Default([](Operation *op) {
                  op->emitOpError()
                      << "hlfir.declare shape expected to be one of: "
                         "fir.shape, fir.shapeshift or fir.shift";
                  return nullptr;
                });

        if (!newShape)
          return failure();

        declareOp.getShapeMutable().assign(newShape);
      }

      for (OpOperand &typeParam : declareOp.getTypeparamsMutable())
        typeParam.assign(zero);

      declareOp.getDummyScopeMutable().clear();
    }

    // We don't actually need the proper initialization, but rather just
    // maintain the basic form of these operands. Generally, we create 1-bit
    // placeholder allocas that we "typecast" to the expected type and replace
    // all uses. Using fir.undefined here instead is not possible because these
    // variables cannot be constants, as that would trigger different codegen
    // for target regions.
    applyChildRewrites(region, childValRewrites, rewriteValues,
                       parentValRewrites);
    for (Value value : rewriteValues) {
      Location loc = value.getLoc();
      Value rewriteValue;
      if (isa_and_present<arith::ConstantOp, fir::AddrOfOp>(
              value.getDefiningOp())) {
        // If it's defined by fir.address_of, then we need to keep that op as
        // well because it might be pointing to a 'declare target' global.
        // Constants can also trigger different codegen paths, so we keep them
        // as well.
        rewriteValue = builder.clone(*value.getDefiningOp())->getResult(0);
      } else if (auto boxCharType =
                     dyn_cast<fir::BoxCharType>(value.getType())) {
        // !fir.boxchar types cannot be directly obtained by converting a
        // !fir.ref<i1>, as they aren't reference types. Since they can appear
        // representing some `target firstprivate` clauses, we need to create
        // a special case here based on creating a placeholder fir.emboxchar op.
        MLIRContext *ctx = &getContext();
        fir::KindTy kind = boxCharType.getKind();
        auto placeholder = builder.create<fir::AllocaOp>(
            loc, fir::CharacterType::getSingleton(ctx, kind));
        auto one = builder.create<arith::ConstantOp>(
            loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
        rewriteValue = builder.create<fir::EmboxCharOp>(loc, boxCharType,
                                                        placeholder, one);
      } else {
        Value placeholder =
            builder.create<fir::AllocaOp>(loc, builder.getI1Type());
        rewriteValue =
            builder.create<fir::ConvertOp>(loc, value.getType(), placeholder);
      }
      value.replaceAllUsesWith(rewriteValue);
    }

    // Move omp.map.info dependencies.
    for (hlfir::DeclareOp declareOp : declareOps)
      declareOp->moveBefore(marker);

    // The box_ref argument of fir.box_offset is expected to be the same value
    // that was passed as var_ptr to the corresponding omp.map.info, so we
    // don't need to handle its defining op here.
    for (fir::BoxOffsetOp boxOffset : boxOffsets)
      boxOffset->moveBefore(marker);

    marker->erase();

    // Move target operations to the end of the new block.
    for (auto targetOp : targetOps)
      std::visit([&block](auto op) { op->moveBefore(&block, block.end()); },
                 targetOp);

    // Add terminator to the new block.
    builder.setInsertionPointToEnd(&block);
    if (funcOp) {
      llvm::SmallVector<Value> returnValues;
      returnValues.reserve(funcOp.getNumResults());
      for (auto type : funcOp.getResultTypes())
        returnValues.push_back(
            builder.create<fir::UndefOp>(funcOp.getLoc(), type));

      builder.create<func::ReturnOp>(funcOp.getLoc(), returnValues);
    } else {
      builder.create<omp::TerminatorOp>(targetDataOp.getLoc());
    }

    // Replace old region (now missing ops) with the new one and remove the
    // temporary operation clone.
    region.takeBody(newOp->getRegion(0));
    newOp->erase();
    return success();
  }
};
} // namespace
