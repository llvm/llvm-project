//===- CIRBasicAliasAnalysis.cpp - Basic CIR Alias Analysis ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/CIRBasicAliasAnalysis.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace llvm;
using namespace cir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

mlir::Value CIRBasicAliasAnalysis::getUnderlyingObject(mlir::Value val) {
  // TODO: Walk through cir.ptr_stride, cir.cast, cir.get_member, etc.
  // to find the root allocation (cir.alloca, cir.global_addr, function args).
  return val;
}

bool CIRBasicAliasAnalysis::areDistinctObjects(mlir::Value lhs, mlir::Value rhs) {
  // Two values are distinct allocations if they originate from different
  // cir.alloca operations (or other allocation ops) in the same function.
  // TODO: Extend to cover global addresses, function arguments with noalias,
  // and heap allocations.
  mlir::Value lhsObj = getUnderlyingObject(lhs);
  mlir::Value rhsObj = getUnderlyingObject(rhs);

  if (lhsObj == rhsObj)
    return false;

  // Different cir.alloca ops in the same function cannot alias.
  if (mlir::isa_and_nonnull<cir::AllocaOp>(lhsObj.getDefiningOp()) &&
      mlir::isa_and_nonnull<cir::AllocaOp>(rhsObj.getDefiningOp()))
    return true;

  return false;
}

//===----------------------------------------------------------------------===//
// CIRBasicAliasAnalysis
//===----------------------------------------------------------------------===//

mlir::AliasResult CIRBasicAliasAnalysis::alias(mlir::Value lhs, mlir::Value rhs) {
  if (lhs == rhs)
    return mlir::AliasResult::MustAlias;

  if (areDistinctObjects(lhs, rhs))
    return mlir::AliasResult::NoAlias;

  // Conservative fallback — the aggregate will try other implementations.
  return mlir::AliasResult::MayAlias;
}

mlir::ModRefResult CIRBasicAliasAnalysis::getModRef(mlir::Operation *op,
                                                    mlir::Value location) {
  // Pure operations (no side effects) neither modify nor reference memory.
  if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
    if (effects.hasNoEffect())
      return mlir::ModRefResult::getNoModRef();

    SmallVector<mlir::MemoryEffects::EffectInstance> effectList;
    effects.getEffects(effectList);

    bool mod = false;
    bool ref = false;
    for (auto &effect : effectList) {
      // Only count effects on the queried location (or unknown location).
      mlir::Value effectVal = effect.getValue();
      if (effectVal && effectVal != location)
        continue;
      if (mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect()))
        mod = true;
      else if (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()))
        ref = true;
    }

    if (!mod && !ref)
      return mlir::ModRefResult::getNoModRef();
    if (mod && !ref)
      return mlir::ModRefResult::getMod();
    if (!mod && ref)
      return mlir::ModRefResult::getRef();
    return mlir::ModRefResult::getModAndRef();
  }

  // Conservative fallback.
  return mlir::ModRefResult::getModAndRef();
}
