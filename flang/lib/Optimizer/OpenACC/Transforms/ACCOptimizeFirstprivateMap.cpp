//===- ACCOptimizeFirstprivateMap.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes firstprivate mapping operations (acc.firstprivate_map).
// For trivial scalar variables, the optimization hoists loads to before the
// compute region, effectively converting the firstprivate copy to a
// pass-by-value pattern. This eliminates the need for runtime copying into
// global memory.
//
// Scalar transformation:
//
//   Before:
//     %decl = fir.declare %alloca : !fir.ref<i32>
//     %fp = acc.firstprivate_map varPtr(%decl) -> !fir.ref<i32>
//     acc.parallel {
//       %val = fir.load %fp : !fir.ref<i32>  // load inside region
//       ...
//     }
//
//   After:
//     %decl = fir.declare %alloca : !fir.ref<i32>
//     %val = fir.load %decl : !fir.ref<i32>  // load hoisted before region
//     acc.parallel {
//       ...  // uses %val directly
//     }
//
// For array descriptors that are read-only within the offload region, the
// optimization reads lower bounds, extents, and byte strides before the region
// and passes them as scalar values. The box is rebuilt inside the region with
// fir.create_box. If all consumers are known metadata queries, the rebuilt box
// uses a null base and the mapping can be erased. Otherwise, the mapping and
// device descriptor load are retained to extract the device base address with
// fir.box_addr.
//
// Descriptor transformation (types omitted):
//
//   Before:
//     %fp = acc.firstprivate_map varPtr(%desc)
//     acc.parallel {
//       %box = fir.load %fp
//       %dims:3 = fir.box_dims %box, %c0
//       ...
//     }
//
//   After:
//     %hostBox = fir.load %desc
//     %hostDims:3 = fir.box_dims %hostBox, %c0
//     acc.parallel {
//       %null = fir.zero_bits
//       %box = fir.create_box %null lbs(%hostDims#0) extents(%hostDims#1)
//                                  strides(%hostDims#2)
//       %dims:3 = fir.box_dims %box, %c0
//       ...
//     }
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/OpenACC/Passes.h"
#include "flang/Optimizer/OpenACC/Support/FIROpenACCUtils.h"
#include "flang/Optimizer/OpenACC/Utils/DescriptorScalarization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace fir::acc {
#define GEN_PASS_DEF_ACCOPTIMIZEFIRSTPRIVATEMAP
#include "flang/Optimizer/OpenACC/Passes.h.inc"
} // namespace fir::acc

using namespace mlir;

namespace {

/// Returns the enclosing offload region interface, or nullptr if not inside
/// one.
static acc::OffloadRegionOpInterface getEnclosingOffloadRegion(Operation *op) {
  return op->getParentOfType<acc::OffloadRegionOpInterface>();
}

/// Returns true if the value is defined by an OpenACC data clause operation.
static bool isDefinedByDataClause(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;
  return acc::getDataClause(defOp).has_value();
}

/// Returns true if the value is defined inside the given offload region.
/// This handles both operation results and block arguments.
static bool isDefinedInsideRegion(Value value,
                                  acc::OffloadRegionOpInterface offloadOp) {
  Region *valueRegion = value.getParentRegion();
  if (!valueRegion)
    return false;
  return offloadOp.getOffloadRegion().isAncestor(valueRegion);
}

/// Returns true if the variable may be optional.
static bool mayBeOptionalVariable(Value var) {
  // Don't strip declare ops - we need to check the optional attribute on them.
  Value originalDef = fir::acc::getOriginalDef(var, /*stripDeclare=*/false);
  if (auto varIface = dyn_cast_or_null<fir::FortranVariableOpInterface>(
          originalDef.getDefiningOp()))
    return varIface.isOptional();
  // If the defining op is an alloca, it's a local variable and not optional.
  if (isa_and_nonnull<fir::AllocaOp, fir::AllocMemOp>(
          originalDef.getDefiningOp()))
    return false;
  // Conservative: if we can't determine, assume it may be optional.
  return true;
}

/// Returns true if `op` is in the then region of an enclosing fir.if whose
/// condition is the presence of `var`. The users of the result of `op` are
/// dominated by it, so they are guarded too.
static bool isGuardedByPresence(Operation *op, Value var) {
  Value varDef = fir::acc::getOriginalDef(var);
  for (auto ifOp = op->getParentOfType<fir::IfOp>(); ifOp;
       op = ifOp, ifOp = ifOp->getParentOfType<fir::IfOp>()) {
    if (!ifOp.getThenRegion().isAncestor(op->getParentRegion()))
      continue;
    auto isPresentOp = ifOp.getCondition().getDefiningOp<fir::IsPresentOp>();
    if (isPresentOp && fir::acc::getOriginalDef(isPresentOp.getVal()) == varDef)
      return true;
  }
  return false;
}

/// Returns true if the type is a reference to a trivial type.
/// Note that this does not allow fir.heap, fir.ptr, or fir.llvm_ptr
/// types - since we would need to check if the load is valid via
/// a null-check to enable the optimization.
static bool isRefToTrivialType(Type type) {
  if (!mlir::isa<fir::ReferenceType>(type))
    return false;
  return fir::isa_trivial(fir::unwrapRefType(type));
}

/// Attempts to hoist loads from accVar to before firstprivateInitOp.
/// If `foldPresenceChecks` is true, the variable is known to be present at
/// firstprivateInitOp, and presence checks of accVar are replaced with true.
/// Returns true if all uses of accVar are loads, or presence checks when
/// `foldPresenceChecks` is true, and the loads were hoisted.
static bool hoistLoads(acc::FirstprivateMapInitialOp firstprivateInitOp,
                       Value var, Value accVar, bool foldPresenceChecks) {
  // Check if all uses are loads or foldable presence checks - only hoist if
  // we can optimize all uses.
  bool allUsesOptimizable =
      llvm::all_of(accVar.getUsers(), [&](Operation *user) {
        return isa<fir::LoadOp>(user) ||
               (foldPresenceChecks && isa<fir::IsPresentOp>(user));
      });
  if (!allUsesOptimizable)
    return false;

  // Replace presence checks with true and hoist all loads before the
  // firstprivate_map operation.
  for (Operation *user : llvm::make_early_inc_range(accVar.getUsers())) {
    if (auto isPresentOp = dyn_cast<fir::IsPresentOp>(user)) {
      OpBuilder builder(isPresentOp);
      Value trueValue = arith::ConstantOp::create(builder, isPresentOp.getLoc(),
                                                  builder.getBoolAttr(true));
      isPresentOp.replaceAllUsesWith(trueValue);
      isPresentOp.erase();
      continue;
    }
    auto loadOp = cast<fir::LoadOp>(user);
    loadOp.getMemrefMutable().assign(var);
    loadOp->moveBefore(firstprivateInitOp);
  }
  return true;
}

static void rebuildDescriptor(fir::LoadOp loadOp, fir::BoxType boxType,
                              ValueRange lowerBounds, ValueRange extents,
                              ValueRange strides) {
  OpBuilder builder(loadOp);
  builder.setInsertionPointAfter(loadOp);
  Location loc = loadOp.getLoc();
  Value box = loadOp.getResult();

  // Use a null base when no user needs the base address so the descriptor load
  // can be removed. If any user may need the base address, retain the load and
  // extract the device base from the original descriptor.
  bool needsBase =
      llvm::any_of(box.getUsers(), fir::acc::mayRequireBaseAddress);
  Type baseType = boxType.getBaseAddressType();
  Value base;
  if (needsBase)
    base = fir::BoxAddrOp::create(builder, loc, baseType, box);
  else
    base = fir::ZeroOp::create(builder, loc, baseType);
  Value rebuilt = fir::CreateBoxOp::create(builder, loc, boxType, base,
                                           lowerBounds, extents, strides);
  box.replaceAllUsesExcept(rebuilt, base.getDefiningOp());
  if (!needsBase)
    loadOp.erase();
}

static bool
scalarizeDescriptor(acc::FirstprivateMapInitialOp firstprivateInitOp,
                    fir::BoxType boxType) {
  Value var = firstprivateInitOp.getVar();

  auto arrayType = fir::acc::getScalarizableArrayType(boxType);
  if (!arrayType)
    return false;

  Value accVar = firstprivateInitOp.getAccVar();

  // All uses must be descriptor loads in the same compute region.
  // Collect the loads.
  acc::OffloadRegionOpInterface offloadOp;
  SmallVector<fir::LoadOp> loads;
  for (Operation *user : accVar.getUsers()) {
    auto loadOp = dyn_cast<fir::LoadOp>(user);
    auto enclosingOffloadOp = getEnclosingOffloadRegion(user);
    if (!loadOp || !enclosingOffloadOp ||
        (offloadOp && offloadOp != enclosingOffloadOp))
      return false;
    offloadOp = enclosingOffloadOp;
    loads.push_back(loadOp);
  }

  // Erase unused loads from the IR and the collected list. If none remain,
  // the firstprivate_map can be erased.
  llvm::erase_if(loads, [](fir::LoadOp loadOp) {
    if (!loadOp->use_empty())
      return false;
    loadOp.erase();
    return true;
  });
  if (loads.empty())
    return true;

  // Read the box dimensions before the offload region.
  OpBuilder builder(offloadOp);
  Location loc = firstprivateInitOp.getLoc();
  Value hostBox = fir::LoadOp::create(builder, loc, var);
  fir::acc::ShapeScalars shape = fir::acc::emitShapeScalars(
      builder, loc, hostBox, arrayType.getDimension());

  for (fir::LoadOp loadOp : loads)
    rebuildDescriptor(loadOp, boxType, shape.lbs, shape.extents, shape.strides);
  return accVar.use_empty();
}

class ACCOptimizeFirstprivateMap
    : public fir::acc::impl::ACCOptimizeFirstprivateMapBase<
          ACCOptimizeFirstprivateMap> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Collect all firstprivate_map ops first to avoid modifying IR during walk.
    llvm::SmallVector<acc::FirstprivateMapInitialOp> firstprivateOps;
    funcOp.walk([&](acc::FirstprivateMapInitialOp op) {
      firstprivateOps.push_back(op);
    });

    llvm::SmallVector<acc::FirstprivateMapInitialOp> opsToErase;

    for (acc::FirstprivateMapInitialOp firstprivateInitOp : firstprivateOps) {
      Value var = firstprivateInitOp.getVar();

      if (auto offloadOp = getEnclosingOffloadRegion(firstprivateInitOp)) {
        // Inside an offload region.
        if (isDefinedByDataClause(var) ||
            isDefinedInsideRegion(var, offloadOp)) {
          // The variable is already mapped or defined locally - just replace
          // uses and erase.
          firstprivateInitOp.getAccVar().replaceAllUsesWith(var);
          opsToErase.push_back(firstprivateInitOp);
        } else {
          // Variable is defined outside - hoist the op out of the region,
          // then apply optimization.
          firstprivateInitOp->moveBefore(offloadOp);
          if (optimizeFirstprivateMapping(firstprivateInitOp))
            opsToErase.push_back(firstprivateInitOp);
        }
      } else {
        // Outside offload region, apply type-restricted optimization
        // to pre-load before the compute region.
        if (optimizeFirstprivateMapping(firstprivateInitOp))
          opsToErase.push_back(firstprivateInitOp);
      }
    }

    for (auto op : opsToErase)
      op.erase();
  }

private:
  /// Returns true if the operation was optimized and can be erased.
  static bool optimizeFirstprivateMapping(
      acc::FirstprivateMapInitialOp firstprivateInitOp) {
    Value var = firstprivateInitOp.getVar();
    Value accVar = firstprivateInitOp.getAccVar();

    // If there are no uses, we can erase the operation.
    if (accVar.use_empty())
      return true;

    // Avoid hoisting optional variables as they may be null and thus not safe
    // to access, unless they are known to be present at this point.
    bool guardedByPresence = false;
    if (mayBeOptionalVariable(var)) {
      guardedByPresence = isGuardedByPresence(firstprivateInitOp, var);
      if (!guardedByPresence)
        return false;
    }

    if (auto refType = dyn_cast<fir::ReferenceType>(var.getType()))
      if (auto boxType = dyn_cast<fir::BoxType>(refType.getEleTy()))
        return scalarizeDescriptor(firstprivateInitOp, boxType);

    // Only optimize references to trivial types.
    if (!isRefToTrivialType(var.getType()))
      return false;

    return hoistLoads(firstprivateInitOp, var, accVar, guardedByPresence);
  }
};

} // namespace

std::unique_ptr<Pass> fir::acc::createACCOptimizeFirstprivateMapPass() {
  return std::make_unique<ACCOptimizeFirstprivateMap>();
}
