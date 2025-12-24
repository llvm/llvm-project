//===- LoopInvariantCodeMotion.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// FIR-specific Loop Invariant Code Motion pass.
/// The pass relies on FIR types and interfaces to prove the safety
/// of hoisting invariant operations out of loop-like operations.
/// It may be run on both HLFIR and FIR representations.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/Support/DebugLog.h"

namespace fir {
#define GEN_PASS_DEF_LOOPINVARIANTCODEMOTION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-licm"

// Temporary engineering option for triaging LICM.
static llvm::cl::opt<bool> disableFlangLICM(
    "disable-flang-licm", llvm::cl::init(false), llvm::cl::Hidden,
    llvm::cl::desc("Disable Flang's loop invariant code motion"));

namespace {

using namespace mlir;

/// The pass tries to hoist loop invariant operations with only
/// MemoryEffects::Read effects (MemoryEffects::Write support
/// may be added later).
/// The safety of hoisting is proven by:
///   * Proving that the loop runs at least one iteration.
///   * Proving that is is always safe to load from this location
///     (see isSafeToHoistLoad() comments below).
struct LoopInvariantCodeMotion
    : fir::impl::LoopInvariantCodeMotionBase<LoopInvariantCodeMotion> {
  void runOnOperation() override;
};

} // namespace

/// 'location' is a memory reference used by a memory access.
/// The type of 'location' defines the data type of the access
/// (e.g. it is considered to be invalid to access 'i64'
/// data using '!fir.ref<i32>`).
/// For the given location, this function returns true iff
/// the Fortran object being accessed is a scalar that
/// may not be OPTIONAL.
///
/// Note that the '!fir.ref<!fir.box<>>' accesses are considered
/// to be scalar, even if the underlying data is an array.
///
/// Note that an access of '!fir.ref<scalar>' may access
/// an array object. For example:
///   real :: x(:)
///   do i=...
///     = x(10)
/// 'x(10)' accesses array 'x', and it may be unsafe to hoist
/// it without proving that '10' is a valid index for the array.
/// The fact that 'x' is not OPTIONAL does not allow hoisting
/// on its own.
static bool isNonOptionalScalar(Value location) {
  while (true) {
    LDBG() << "Checking location:\n" << location;
    Type dataType = fir::unwrapRefType(location.getType());
    if (!isa<fir::BaseBoxType>(location.getType()) &&
        (!dataType ||
         (!isa<fir::BaseBoxType>(dataType) && !fir::isa_trivial(dataType) &&
          !fir::isa_derived(dataType)))) {
      LDBG() << "Failure: data access is not scalar";
      return false;
    }
    Operation *defOp = location.getDefiningOp();
    if (!defOp) {
      LDBG() << "Failure: no defining operation";
      return false;
    }
    if (auto varIface = dyn_cast<fir::FortranVariableOpInterface>(defOp)) {
      bool result = !varIface.isOptional();
      if (result)
        LDBG() << "Success: is non optional scalar";
      else
        LDBG() << "Failure: is not non optional scalar";
      return result;
    }
    if (auto viewIface = dyn_cast<fir::FortranObjectViewOpInterface>(defOp)) {
      location = viewIface.getViewSource(cast<OpResult>(location));
    } else {
      LDBG() << "Failure: unknown operation:\n" << *defOp;
      return false;
    }
  }
}

/// Returns true iff it is safe to hoist the given load-like operation 'op',
/// which access given memory 'locations', out of the operation 'loopLike'.
/// The current safety conditions are:
///   * The loop runs at least one iteration, OR
///   * all the accessed locations are inside scalar non-OPTIONAL
///     Fortran objects (Fortran descriptors are considered to be scalars).
static bool isSafeToHoistLoad(Operation *op, ArrayRef<Value> locations,
                              LoopLikeOpInterface loopLike,
                              AliasAnalysis &aliasAnalysis) {
  for (Value location : locations)
    if (aliasAnalysis.getModRef(loopLike.getOperation(), location)
            .isModAndRef()) {
      LDBG() << "Failure: reads location:\n"
             << location << "\nwhich is modified inside the loop";
      return false;
    }

  // Check that it is safe to read from all the locations before the loop.
  std::optional<llvm::APInt> tripCount = loopLike.getStaticTripCount();
  if (tripCount && !tripCount->isZero()) {
    // Loop executes at least one iteration, so it is safe to hoist.
    LDBG() << "Success: loop has non-zero iterations";
    return true;
  }

  // Check whether the access must always be valid.
  return llvm::all_of(
      locations, [&](Value location) { return isNonOptionalScalar(location); });
  // TODO: consider hoisting under condition of the loop's trip count
  // being non-zero.
}

/// Returns true iff the given 'op' is a load-like operation,
/// and it can be hoisted out of 'loopLike' operation.
static bool canHoistLoad(Operation *op, LoopLikeOpInterface loopLike,
                         AliasAnalysis &aliasAnalysis) {
  LDBG() << "Checking operation:\n" << *op;
  if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    effectInterface.getEffects(effects);
    if (effects.empty()) {
      LDBG() << "Failure: not a load";
      return false;
    }
    llvm::SetVector<Value> locations;
    for (const MemoryEffects::EffectInstance &effect : effects) {
      Value location = effect.getValue();
      if (!isa<MemoryEffects::Read>(effect.getEffect())) {
        LDBG() << "Failure: has unsupported effects";
        return false;
      } else if (!location) {
        LDBG() << "Failure: reads from unknown location";
        return false;
      }
      locations.insert(location);
    }
    return isSafeToHoistLoad(op, locations.getArrayRef(), loopLike,
                             aliasAnalysis);
  }
  LDBG() << "Failure: has unknown effects";
  return false;
}

void LoopInvariantCodeMotion::runOnOperation() {
  if (disableFlangLICM) {
    LDBG() << "Skipping [HL]FIR LoopInvariantCodeMotion()";
    return;
  }

  LDBG() << "Enter [HL]FIR LoopInvariantCodeMotion()";

  auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
  aliasAnalysis.addAnalysisImplementation(fir::AliasAnalysis{});

  std::function<bool(Operation *, LoopLikeOpInterface loopLike)>
      shouldMoveOutOfLoop = [&](Operation *op, LoopLikeOpInterface loopLike) {
        if (isPure(op)) {
          LDBG() << "Pure operation: " << *op;
          return true;
        }

        // Handle RecursivelySpeculatable operations that have
        // RecursiveMemoryEffects by checking if all their
        // nested operations can be hoisted.
        auto iface = dyn_cast<ConditionallySpeculatable>(op);
        if (iface && iface.getSpeculatability() ==
                         Speculation::RecursivelySpeculatable) {
          if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
            LDBG() << "Checking recursive operation:\n" << *op;
            llvm::SmallVector<Operation *> nestedOps;
            for (Region &region : op->getRegions())
              for (Block &block : region)
                for (Operation &nestedOp : block)
                  nestedOps.push_back(&nestedOp);

            bool result = llvm::all_of(nestedOps, [&](Operation *nestedOp) {
              return shouldMoveOutOfLoop(nestedOp, loopLike);
            });
            LDBG() << "Recursive operation can" << (result ? "" : "not")
                   << " be hoisted";

            // If nested operations cannot be hoisted, there is nothing
            // else to check. Also if the operation itself does not have
            // any memory effects, we can return the result now.
            // Otherwise, we have to check the operation itself below.
            if (!result || !isa<MemoryEffectOpInterface>(op))
              return result;
          }
        }
        return canHoistLoad(op, loopLike, aliasAnalysis);
      };

  getOperation()->walk([&](LoopLikeOpInterface loopLike) {
    moveLoopInvariantCode(
        loopLike.getLoopRegions(),
        /*isDefinedOutsideRegion=*/
        [&](Value value, Region *) {
          return loopLike.isDefinedOutsideOfLoop(value);
        },
        /*shouldMoveOutOfRegion=*/
        [&](Operation *op, Region *) {
          return shouldMoveOutOfLoop(op, loopLike);
        },
        /*moveOutOfRegion=*/
        [&](Operation *op, Region *) { loopLike.moveOutOfLoop(op); });
  });

  LDBG() << "Exit [HL]FIR LoopInvariantCodeMotion()";
}
