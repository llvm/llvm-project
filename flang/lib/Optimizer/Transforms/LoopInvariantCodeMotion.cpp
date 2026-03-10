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
#include "flang/Optimizer/Dialect/FIROperationMoveOpInterface.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/TypeSwitch.h"
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
      // If this is a function argument
      auto blockArg = cast<BlockArgument>(location);
      Block *block = blockArg.getOwner();
      if (block && block->isEntryBlock())
        if (auto funcOp =
                dyn_cast_if_present<FunctionOpInterface>(block->getParentOp()))
          if (!funcOp.getArgAttrOfType<UnitAttr>(blockArg.getArgNumber(),
                                                 fir::getOptionalAttrName())) {
            LDBG() << "Success: is non optional scalar dummy";
            return true;
          }

      LDBG() << "Failure: no defining operation";
      return false;
    }

    // Scalars "defined" by fir.alloca and fir.address_of
    // are present.
    if (isa<fir::AllocaOp, fir::AddrOfOp>(defOp)) {
      LDBG() << "Success: is non optional scalar";
      return true;
    }

    if (auto varIface = dyn_cast<fir::FortranVariableOpInterface>(defOp)) {
      if (varIface.isOptional()) {
        // The variable is optional, so do not look further.
        // Note that it is possible to deduce that the optional
        // is actually present, but we are not doing it now.
        LDBG() << "Failure: is optional";
        return false;
      }

      // In case of MLIR inlining and ASSOCIATE an [hl]fir.declare
      // may declare a scalar variable that is actually a "view"
      // of an array element. Originally, such [hl]fir.declare
      // would be located inside the loop preventing the hoisting.
      // But if we decide to hoist such [hl]fir.declare in future,
      // we cannot rely on their attributes/types.
      // Use reliable checks based on the variable storage.

      // If the variable has storage specifier (e.g. it is a member
      // of COMMON, etc.), we can rely that the storage is present,
      // and we can also rely on its FortranVariableOpInterface
      // definition type (which is a scalar due to previous checks).
      if (auto storageIface =
              dyn_cast<fir::FortranVariableStorageOpInterface>(defOp))
        if (Value storage = storageIface.getStorage()) {
          LDBG() << "Success: is scalar with existing storage";
          return true;
        }

      // TODO: we can probably use FIR AliasAnalysis' getSource()
      // method to identify the storage in more cases.
      Value memref = llvm::TypeSwitch<Operation *, Value>(defOp)
                         .Case<fir::DeclareOp, hlfir::DeclareOp>(
                             [](auto op) { return op.getMemref(); })
                         .Default([](auto) { return nullptr; });

      if (memref)
        return isNonOptionalScalar(memref);

      LDBG() << "Failure: cannot reason about variable storage";
      return false;
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
    if (!fir::canMoveOutOf(loopLike, nullptr)) {
      LDBG() << "Cannot hoist anything out of loop operation: ";
      LDBG_OS([&](llvm::raw_ostream &os) {
        loopLike->print(os, OpPrintingFlags().skipRegions());
      });
      return;
    }
    // We always hoist operations to the parent operation of the loopLike.
    // Check that the parent operation allows the hoisting, e.g.
    // omp::LoopWrapperInterface operations assume tight nesting
    // of the inner maybe loop-like operations, so hoisting
    // to such a parent would be invalid. We rely on
    // fir::canMoveFromDescendant() to identify whether the hoisting
    // is allowed.
    Operation *parentOp = loopLike->getParentOp();
    if (!parentOp) {
      LDBG() << "Skipping top-level loop-like operation?";
      return;
    } else if (!fir::canMoveFromDescendant(parentOp, loopLike, nullptr)) {
      LDBG() << "Cannot hoist anything into operation: ";
      LDBG_OS([&](llvm::raw_ostream &os) {
        parentOp->print(os, OpPrintingFlags().skipRegions());
      });
      return;
    }
    moveLoopInvariantCode(
        loopLike.getLoopRegions(),
        /*isDefinedOutsideRegion=*/
        [&](Value value, Region *) {
          return loopLike.isDefinedOutsideOfLoop(value);
        },
        /*shouldMoveOutOfRegion=*/
        [&](Operation *op, Region *) {
          if (!fir::canMoveOutOf(loopLike, op)) {
            LDBG() << "Cannot hoist " << *op << " out of the loop";
            return false;
          }
          if (!fir::canMoveFromDescendant(parentOp, loopLike, op)) {
            LDBG() << "Cannot hoist " << *op << " into the parent of the loop";
            return false;
          }
          return shouldMoveOutOfLoop(op, loopLike);
        },
        /*moveOutOfRegion=*/
        [&](Operation *op, Region *) { loopLike.moveOutOfLoop(op); });
  });

  LDBG() << "Exit [HL]FIR LoopInvariantCodeMotion()";
}
