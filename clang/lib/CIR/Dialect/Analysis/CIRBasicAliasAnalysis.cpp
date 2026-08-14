//===- CIRBasicAliasAnalysis.cpp - Basic CIR Alias Analysis ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/CIRBasicAliasAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "cir-basic-alias-analysis"

using namespace llvm;
using namespace cir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

mlir::Value CIRBasicAliasAnalysis::getUnderlyingObject(mlir::Value val) {
  LDBG() << "Getting underlying object for: " << val;

  // TODO: Walk through cir.ptr_stride, cir.cast, cir.get_member, etc.
  // to find the root allocation (cir.alloca, cir.global_addr, function args).
  LDBG() << "Not yet implemented";
  return val;
}

bool CIRBasicAliasAnalysis::areDistinctObjects(mlir::Value lhs,
                                               mlir::Value rhs) {
  LDBG() << "Checking if " << lhs << " and " << rhs << " are distinct objects";

  // Two values are distinct allocations if they originate from different
  // cir.alloca operations (or other allocation ops) in the same function.
  // TODO: Extend to cover global addresses, function arguments with noalias,
  // and heap allocations.
  mlir::Value lhsObj = getUnderlyingObject(lhs);
  mlir::Value rhsObj = getUnderlyingObject(rhs);

  if (lhsObj == rhsObj) {
    LDBG() << "Identical values, not distinct";
    return false;
  }

  // Different cir.alloca ops in the same function cannot alias.
  if (mlir::isa_and_nonnull<cir::AllocaOp>(lhsObj.getDefiningOp()) &&
      mlir::isa_and_nonnull<cir::AllocaOp>(rhsObj.getDefiningOp())) {
    LDBG() << "Different cir.alloca ops in the same function, distinct";
    return true;
  }

  LDBG() << "Conservative fallback, not distinct";
  return false;
}

//===----------------------------------------------------------------------===//
// CIRBasicAliasAnalysis
//===----------------------------------------------------------------------===//

mlir::AliasResult CIRBasicAliasAnalysis::alias(mlir::Value lhs,
                                               mlir::Value rhs) {
  LDBG() << "Checking alias between: " << lhs << " and " << rhs;

  if (lhs == rhs) {
    LDBG() << "Trivial alias between identical values";
    return mlir::AliasResult::MustAlias;
  }

  if (areDistinctObjects(lhs, rhs)) {
    LDBG() << "No alias between distinct objects";
    return mlir::AliasResult::NoAlias;
  }

  // Conservative fallback — the aggregate will try other implementations.
  LDBG() << "Conservative fallback, may alias";
  return mlir::AliasResult::MayAlias;
}

mlir::ModRefResult CIRBasicAliasAnalysis::getModRef(mlir::Operation *op,
                                                    mlir::Value location) {
  LDBG() << "getModRef: "
         << mlir::OpWithFlags(op, mlir::OpPrintingFlags().skipRegions())
         << " on location " << location;

  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!effects) {
    LDBG() << "No memory effect interface, returning ModAndRef";
    return mlir::ModRefResult::getModAndRef();
  }

  SmallVector<mlir::MemoryEffects::EffectInstance> effectList;
  effects.getEffects(effectList);

  auto classifyEffect = [location, this](
                            const mlir::MemoryEffects::EffectInstance &effect) {
    if (mlir::isa<mlir::MemoryEffects::Allocate>(effect.getEffect())) {
      LDBG() << "Skipping allocate effect";
      return mlir::ModRefResult::getNoModRef();
    }

    mlir::AliasResult aliasResult = mlir::AliasResult::MayAlias;
    if (mlir::Value affectedLocation = effect.getValue()) {
      LDBG() << "    Checking alias between affected location "
             << affectedLocation << " and query location " << location;
      aliasResult = alias(affectedLocation, location);
      LDBG() << "    Alias result: "
             << (aliasResult.isMust() ? "MustAlias"
                 : aliasResult.isNo() ? "NoAlias"
                                      : "MayAlias");
    } else {
      // An effect on a non-addressable resource cannot affect a
      // pointer-based location.
      if (!effect.getResource()->isAddressable()) {
        LDBG() << "    Effect on non-addressable resource '"
               << effect.getResource()->getName() << "', skipping (NoAlias)";
        aliasResult = mlir::AliasResult::NoAlias;
      } else {
        LDBG() << "    No effect value, assuming MayAlias";
      }
    }

    // If the affected location doesn't alias with the query location,
    // ignore this effect.
    if (aliasResult.isNo()) {
      LDBG() << "No alias with affected location";
      return mlir::ModRefResult::getNoModRef();
    }

    // TODO: Consider whether Free should be NoModRef.
    if (mlir::isa<mlir::MemoryEffects::Free>(effect.getEffect())) {
      LDBG() << "Skipping free effect";
      return mlir::ModRefResult::getModAndRef();
    }

    if (mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
      LDBG() << "Write effect, adding Mod";
      return mlir::ModRefResult::getMod();
    }

    if (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect())) {
      LDBG() << "Read effect, adding Ref";
      return mlir::ModRefResult::getRef();
    }

    LDBG() << "Unexpected memory effect: " << effect.getEffect();
    return mlir::ModRefResult::getNoModRef();
  };

  return llvm::accumulate(llvm::map_range(effectList, classifyEffect),
                          mlir::ModRefResult::getNoModRef(),
                          [](mlir::ModRefResult lhs, mlir::ModRefResult rhs) {
                            return lhs.merge(rhs);
                          });
}
