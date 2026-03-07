//===- HostOpFiltering.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements transforms to filter out host-only operations from
// any remaining host functions when compiling for a target device.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace flangomp {
#define GEN_PASS_DEF_HOSTOPFILTERINGPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

using namespace mlir;

/// Add an operation to one of the output sets to be later rewritten.
template <typename OpTy>
static void collectRewrite(OpTy op, llvm::SetVector<OpTy> &rewrites) {
  rewrites.insert(op);
}

/// Add an \c omp.map.info operation and all its members recursively to the
/// output set to be later rewritten.
///
/// Dependencies across \c omp.map.info are maintained by ensuring dependencies
/// are added to the output sets before operations based on them.
template <>
void collectRewrite(omp::MapInfoOp mapOp,
                    llvm::SetVector<omp::MapInfoOp> &rewrites) {
  for (Value member : mapOp.getMembers())
    collectRewrite(cast<omp::MapInfoOp>(member.getDefiningOp()), rewrites);

  rewrites.insert(mapOp);
}

/// Add the given value to a sorted set if it should be replaced by a
/// placeholder when used as an operand that must remain for the device.
///
/// Values that are block arguments of \c func.func operations are skipped,
/// since they will still be available after all rewrites are completed.
static void collectRewrite(Value value, llvm::SetVector<Value> &rewrites) {
  if ((isa<BlockArgument>(value) &&
       isa<func::FuncOp>(
           cast<BlockArgument>(value).getOwner()->getParentOp())) ||
      rewrites.contains(value))
    return;

  rewrites.insert(value);
}

namespace {
class HostOpFilteringPass
    : public flangomp::impl::HostOpFilteringPassBase<HostOpFilteringPass> {
public:
  HostOpFilteringPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OpBuilder opBuilder(context);
    auto op = dyn_cast<omp::OffloadModuleInterface>(getOperation());
    if (!op || !op.getIsTargetDevice())
      return;

    op->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
      omp::DeclareTargetDeviceType declareType =
          omp::DeclareTargetDeviceType::host;
      auto declareTargetOp =
          dyn_cast<omp::DeclareTargetInterface>(funcOp.getOperation());
      if (declareTargetOp && declareTargetOp.isDeclareTarget())
        declareType = declareTargetOp.getDeclareTargetDeviceType();

      // Only process host function definitions.
      if (funcOp.isExternal() ||
          declareType != omp::DeclareTargetDeviceType::host)
        return WalkResult::advance();

      if (failed(rewriteHostFunction(funcOp))) {
        funcOp.emitOpError() << "could not filter host-only operations";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }

private:
  /// Rewrite the given host device function containing \c omp.target
  /// operations, to remove host-only operations that are not used by device
  /// codegen.
  ///
  /// It is based on the expected form of the MLIR module as produced by Flang
  /// lowering, after HLFIR to FIR lowering, and it performs the following
  /// mutations:
  ///   - Replace all values returned by the function with \c fir.undefined.
  ///   - \c omp.target operations are moved to the end of the function. If they
  ///     are nested inside of any other operations, they are hoisted out of
  ///     them.
  ///   - \c depend, \c device and \c if clauses are removed from these target
  ///     functions. Values used to initialize other clauses are replaced by
  ///     placeholders as follows:
  ///     - Values defined by block arguments are replaced by placeholders only
  ///       if they are not attached to the parent \c func.func operation. In
  ///       that case, they are passed unmodified.
  ///     - \c arith.constant and \c fir.address_of ops are maintained.
  ///     - Values of type \c fir.boxchar are replaced with a combination of
  ///       \c fir.alloca for a single bit and a \c fir.emboxchar.
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
  ///     - \c var_ptr can be defined by a \c fir.declare which is also
  ///       preserved. Its \c memref argument is replaced by a placeholder or
  ///       maintained, similarly to non-map clauses of target operations
  ///       described above. If it has \c shape or \c typeparams arguments, they
  ///       are replaced by applicable constants. \c dummy_scope arguments
  ///       are discarded.
  ///   - Every other operation not located inside of an \c omp.target is
  ///     removed.
  LogicalResult rewriteHostFunction(func::FuncOp funcOp) {
    Region &region = funcOp.getRegion();

    // Collect target operations inside of the function.
    llvm::SmallVector<omp::TargetOp> targetOps;
    region.walk<WalkOrder::PreOrder>([&](Operation *op) {
      // Skip the inside of omp.target regions, since these contain device code.
      if (auto targetOp = dyn_cast<omp::TargetOp>(op)) {
        targetOps.push_back(targetOp);
        return WalkResult::skip();
      }

      // Replace omp.target_data entry block argument uses with the value used
      // to initialize the associated omp.map.info operation. This way,
      // references are still valid once the omp.target operation has been
      // extracted out of the omp.target_data region.
      if (auto targetDataOp = dyn_cast<omp::TargetDataOp>(op)) {
        llvm::SmallVector<std::pair<Value, BlockArgument>> argPairs;
        cast<omp::BlockArgOpenMPOpInterface>(*targetDataOp)
            .getBlockArgsPairs(argPairs);
        for (auto [operand, blockArg] : argPairs) {
          auto mapInfo = cast<omp::MapInfoOp>(operand.getDefiningOp());
          Value varPtr = mapInfo.getVarPtr();

          // If the var_ptr operand of the omp.map.info op defining this entry
          // block argument is a fir.declare, the uses of all users of that
          // entry block argument that are themselves fir.declare are replaced
          // by the value produced by the outer one.
          //
          // This prevents this pass from producing chains of fir.declare of the
          // type:
          // %0 = ...
          // %1 = fir.declare %0
          // %2 = fir.declare %1...
          // %3 = omp.map.info var_ptr(%2 ...
          if (auto outerDeclare = varPtr.getDefiningOp<fir::DeclareOp>())
            for (Operation *user : blockArg.getUsers())
              if (isa<fir::DeclareOp>(user))
                user->replaceAllUsesWith(outerDeclare);

          // All remaining uses of the entry block argument are replaced with
          // the var_ptr initialization value.
          blockArg.replaceAllUsesWith(varPtr);
        }
      }
      return WalkResult::advance();
    });

    // Make a temporary clone of the parent operation with an empty region,
    // and update all references to entry block arguments to those of the new
    // region. Users will later either be moved to the new region or deleted
    // when the original region is replaced by the new.
    OpBuilder builder(&getContext());
    builder.setInsertionPointAfter(funcOp);
    Operation *newOp = builder.cloneWithoutRegions(funcOp);
    Block &block = newOp->getRegion(0).emplaceBlock();

    llvm::SmallVector<Location> locs;
    locs.reserve(region.getNumArguments());
    llvm::transform(region.getArguments(), std::back_inserter(locs),
                    [](const BlockArgument &arg) { return arg.getLoc(); });
    block.addArguments(region.getArgumentTypes(), locs);

    for (auto [oldArg, newArg] :
         llvm::zip_equal(region.getArguments(), block.getArguments()))
      oldArg.replaceAllUsesWith(newArg);

    // Collect omp.map.info ops while satisfying interdependencies and remove
    // operands that aren't used by target device codegen.
    //
    // This logic must be updated whenever operands to omp.target change.
    llvm::SetVector<Value> rewriteValues;
    llvm::SetVector<omp::MapInfoOp> mapInfos;
    for (omp::TargetOp targetOp : targetOps) {
      assert(targetOp.getHostEvalVars().empty() &&
             "unexpected host_eval in target device module");

      // Variables unused by the device.
      targetOp.getDependVarsMutable().clear();
      targetOp.setDependKindsAttr(nullptr);
      targetOp.getDeviceMutable().clear();
      targetOp.getIfExprMutable().clear();

      // TODO: Clear some of these operands rather than rewriting them,
      // depending on whether they are needed by device codegen once support for
      // them is fully implemented.
      for (Value allocVar : targetOp.getAllocateVars())
        collectRewrite(allocVar, rewriteValues);
      for (Value allocVar : targetOp.getAllocatorVars())
        collectRewrite(allocVar, rewriteValues);
      for (Value inReduction : targetOp.getInReductionVars())
        collectRewrite(inReduction, rewriteValues);
      for (Value isDevPtr : targetOp.getIsDevicePtrVars())
        collectRewrite(isDevPtr, rewriteValues);
      for (Value mapVar : targetOp.getHasDeviceAddrVars())
        collectRewrite(cast<omp::MapInfoOp>(mapVar.getDefiningOp()), mapInfos);
      for (Value mapVar : targetOp.getMapVars())
        collectRewrite(cast<omp::MapInfoOp>(mapVar.getDefiningOp()), mapInfos);
      for (Value privateVar : targetOp.getPrivateVars())
        collectRewrite(privateVar, rewriteValues);
      for (Value threadLimit : targetOp.getThreadLimitVars())
        collectRewrite(threadLimit, rewriteValues);
    }

    // Move omp.map.info ops to the new block and collect dependencies.
    llvm::SetVector<fir::DeclareOp> declareOps;
    llvm::SetVector<fir::BoxOffsetOp> boxOffsets;
    for (omp::MapInfoOp mapOp : mapInfos) {
      if (auto declareOp = dyn_cast_if_present<fir::DeclareOp>(
              mapOp.getVarPtr().getDefiningOp()))
        collectRewrite(declareOp, declareOps);
      else
        collectRewrite(mapOp.getVarPtr(), rewriteValues);

      if (Value varPtrPtr = mapOp.getVarPtrPtr()) {
        if (auto boxOffset = llvm::dyn_cast_if_present<fir::BoxOffsetOp>(
                varPtrPtr.getDefiningOp()))
          collectRewrite(boxOffset, boxOffsets);
        else
          return mapOp->emitOpError() << "var_ptr_ptr rewrite only supported "
                                         "if defined by fir.box_offset";
      }

      // Bounds are not used during target device codegen.
      mapOp.getBoundsMutable().clear();
      mapOp->moveBefore(&block, block.end());
    }

    // Create a temporary marker to simplify the op moving process below.
    builder.setInsertionPointToStart(&block);
    auto marker = fir::UndefOp::create(builder, builder.getUnknownLoc(),
                                       builder.getNoneType());
    builder.setInsertionPoint(marker);

    // Handle dependencies of fir.declare ops.
    for (fir::DeclareOp declareOp : declareOps) {
      // FIR: memref, shape, typeparams, dummy_scope, storage, storage_offset,
      //      uniq_name, fortran_attrs, data_attr, dummy_arg_no
      // HLFIR: skip_rebox
      collectRewrite(declareOp.getMemref(), rewriteValues);

      if (declareOp.getStorage())
        collectRewrite(declareOp.getStorage(), rewriteValues);

      // Shape and typeparams aren't needed for target device codegen, but
      // removing them would break verifiers.
      Value zero;
      if (declareOp.getShape() || !declareOp.getTypeparams().empty())
        zero = arith::ConstantOp::create(builder, declareOp.getLoc(),
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
                  return fir::ShapeOp::create(builder, op.getLoc(), extents);
                })
                .Case([&](fir::ShapeShiftOp op) {
                  auto type = fir::ShapeShiftType::get(op.getContext(),
                                                       extents.size() / 2);
                  return fir::ShapeShiftOp::create(builder, op.getLoc(), type,
                                                   extents);
                })
                .Case([&](fir::ShiftOp op) {
                  auto type =
                      fir::ShiftType::get(op.getContext(), extents.size());
                  return fir::ShiftOp::create(builder, op.getLoc(), type,
                                              extents);
                })
                .Default([](Operation *op) {
                  op->emitOpError()
                      << "fir.declare shape expected to be one of: "
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
        auto placeholder = fir::AllocaOp::create(
            builder, loc, fir::CharacterType::getSingleton(ctx, kind));
        auto one = arith::ConstantOp::create(builder, loc, builder.getI32Type(),
                                             builder.getI32IntegerAttr(1));
        rewriteValue = fir::EmboxCharOp::create(builder, loc, boxCharType,
                                                placeholder, one);
      } else {
        Value placeholder =
            fir::AllocaOp::create(builder, loc, builder.getI1Type());
        rewriteValue =
            fir::ConvertOp::create(builder, loc, value.getType(), placeholder);
      }
      value.replaceAllUsesWith(rewriteValue);
    }

    // Move omp.map.info dependencies.
    for (fir::DeclareOp declareOp : declareOps)
      declareOp->moveBefore(marker);

    // The box_ref argument of fir.box_offset is expected to be the same value
    // that was passed as var_ptr to the corresponding omp.map.info, so we don't
    // need to handle its defining op here.
    for (fir::BoxOffsetOp boxOffset : boxOffsets)
      boxOffset->moveBefore(marker);

    marker->erase();

    // Move target operations to the end of the new block.
    for (omp::TargetOp targetOp : targetOps)
      targetOp->moveBefore(&block, block.end());

    // Add terminator to the new block.
    builder.setInsertionPointToEnd(&block);
    llvm::SmallVector<Value> returnValues;
    returnValues.reserve(funcOp.getNumResults());
    for (auto type : funcOp.getResultTypes())
      returnValues.push_back(
          fir::UndefOp::create(builder, funcOp.getLoc(), type));

    func::ReturnOp::create(builder, funcOp.getLoc(), returnValues);

    // Replace old region (now missing ops) with the new one and remove the
    // temporary operation clone.
    region.takeBody(newOp->getRegion(0));
    newOp->erase();
    return success();
  }
};
} // namespace
