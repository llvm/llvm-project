//===- ACCOptimizeFirstprivateMap.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes firstprivate mapping operations (acc.firstprivate_map).
// The optimization hoists loads from the firstprivate variable to before the
// compute region, effectively converting the firstprivate copy to a
// pass-by-value pattern. This eliminates the need for runtime copying into
// global memory.
//
// Example transformation:
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
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/OpenACC/Passes.h"
#include "flang/Optimizer/OpenACC/Support/FIROpenACCUtils.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
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
/// Returns true if all uses of accVar are loads and they were hoisted.
static bool hoistLoads(acc::FirstprivateMapInitialOp firstprivateInitOp,
                       Value var, Value accVar) {
  // Check if all uses are loads - only hoist if we can optimize all uses.
  bool allLoads = llvm::all_of(accVar.getUsers(), [](Operation *user) {
    return isa<fir::LoadOp>(user);
  });
  if (!allLoads)
    return false;

  // Hoist all loads before the firstprivate_map operation.
  for (Operation *user : llvm::make_early_inc_range(accVar.getUsers())) {
    auto loadOp = cast<fir::LoadOp>(user);
    loadOp.getMemrefMutable().assign(var);
    loadOp->moveBefore(firstprivateInitOp);
  }
  return true;
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

    // Only optimize references to trivial types.
    if (!isRefToTrivialType(var.getType()))
      return false;

    // Avoid hoisting optional variables as they may be
    // null and thus not safe to access.
    if (mayBeOptionalVariable(var))
      return false;

    return hoistLoads(firstprivateInitOp, var, accVar);
  }
};

} // namespace

std::unique_ptr<Pass> fir::acc::createACCOptimizeFirstprivateMapPass() {
  return std::make_unique<ACCOptimizeFirstprivateMap>();
}
