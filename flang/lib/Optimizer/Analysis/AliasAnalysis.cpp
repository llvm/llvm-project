//===- AliasAnalysis.cpp - Alias Analysis for FIR  ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AliasAnalysis: alias
//===----------------------------------------------------------------------===//

namespace fir {
AliasResult AliasAnalysis::alias(Value lhs, Value rhs) {
  // This is for now a mock analysis
  if (lhs == rhs) {
    return AliasResult::MustAlias;
  }
  return AliasResult::MayAlias;
}

//===----------------------------------------------------------------------===//
// AliasAnalysis: getModRef
//===----------------------------------------------------------------------===//

/// This is mostly inspired by MLIR::LocalAliasAnalysis with 2 notable
/// differences 1) Regions are not handled here but will be handled by a data
/// flow analysis to come 2) Allocate and Free effects are considered modifying
ModRefResult AliasAnalysis::getModRef(Operation *op, Value location) {
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface)
    return ModRefResult::getModAndRef();

  // Build a ModRefResult by merging the behavior of the effects of this
  // operation.
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);

  ModRefResult result = ModRefResult::getNoModRef();
  for (const MemoryEffects::EffectInstance &effect : effects) {

    // Check for an alias between the effect and our memory location.
    AliasResult aliasResult = AliasResult::MayAlias;
    if (Value effectValue = effect.getValue())
      aliasResult = alias(effectValue, location);

    // If we don't alias, ignore this effect.
    if (aliasResult.isNo())
      continue;

    // Merge in the corresponding mod or ref for this effect.
    if (isa<MemoryEffects::Read>(effect.getEffect())) {
      result = result.merge(ModRefResult::getRef());
    } else {
      result = result.merge(ModRefResult::getMod());
    }
    if (result.isModAndRef())
      break;
  }
  return result;
}
} // namespace fir
