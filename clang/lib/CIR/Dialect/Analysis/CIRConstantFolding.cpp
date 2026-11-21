//===- CIRConstantFolding.cpp - Fold CIR values without rewriting ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/CIRConstantFolding.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace cir;

static mlir::Attribute foldToConstant(mlir::Value value, unsigned depth);

/// Fold result `index` of `branch` to the constant every value flowing into it
/// agrees on, given its operands folded to `operands`.
static mlir::Attribute
foldRegionBranchResult(mlir::RegionBranchOpInterface branch, unsigned index,
                       llvm::ArrayRef<mlir::Attribute> operands,
                       unsigned depth) {
  // The regions control may reach: those the operands let it enter, and those
  // reached from them.
  llvm::SmallVector<mlir::RegionSuccessor> worklist;
  branch.getEntrySuccessorRegions(operands, worklist);
  llvm::SmallPtrSet<mlir::Region *, 4> reachable;
  bool mayEnterNoRegion = false;
  while (!worklist.empty()) {
    mlir::RegionSuccessor successor = worklist.pop_back_val();
    if (successor.isOperation()) {
      mayEnterNoRegion = true;
      continue;
    }
    mlir::Region *region = successor.getSuccessor();
    if (!reachable.insert(region).second)
      continue;
    for (mlir::Block &block : *region) {
      if (!block.mightHaveTerminator())
        continue;
      if (auto terminator =
              mlir::dyn_cast<mlir::RegionBranchTerminatorOpInterface>(
                  block.getTerminator()))
        branch.getSuccessorRegions(terminator, worklist);
    }
  }

  mlir::RegionSuccessor parent(branch.getOperation());
  llvm::SmallVector<mlir::RegionBranchPoint> predecessors;
  branch.getPredecessors(parent, predecessors);
  mlir::Attribute folded;
  for (mlir::RegionBranchPoint predecessor : predecessors) {
    std::optional<mlir::OperandRange> values;
    if (predecessor.isParent()) {
      if (mayEnterNoRegion)
        values = branch.getEntrySuccessorOperands(parent);
    } else {
      mlir::RegionBranchTerminatorOpInterface terminator =
          predecessor.getTerminatorPredecessorOrNull();
      if (reachable.contains(terminator->getParentRegion()))
        values = terminator.getSuccessorOperands(parent);
    }
    if (!values)
      continue;
    if (index >= values->size())
      return {};
    mlir::Attribute attr = foldToConstant((*values)[index], depth + 1);
    if (!attr || (folded && attr != folded))
      return {};
    folded = attr;
  }
  return folded;
}

// TODO: Look into sharing this logic with MLIR's dataflow constant
// propagation, which is similar but solves whole functions: too much compile
// time when only a few values are needed.
static mlir::Attribute foldToConstant(mlir::Value value, unsigned depth) {
  mlir::Attribute attr;
  if (mlir::matchPattern(value, mlir::m_Constant(&attr)))
    return attr;

  constexpr unsigned maxDepth = 8;
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result || depth >= maxDepth)
    return {};

  mlir::Operation *def = result.getOwner();
  llvm::SmallVector<mlir::Attribute> operands;
  for (mlir::Value operand : def->getOperands())
    operands.push_back(foldToConstant(operand, depth + 1));

  if (auto branch = mlir::dyn_cast<mlir::RegionBranchOpInterface>(def))
    return foldRegionBranchResult(branch, result.getResultNumber(), operands,
                                  depth);

  if (def->getNumRegions() != 0 || !mlir::isMemoryEffectFree(def) ||
      llvm::is_contained(operands, mlir::Attribute()))
    return {};

  // CIR folders never update their operation in place, so this leaves the IR
  // unchanged.
  llvm::SmallVector<mlir::OpFoldResult> results;
  if (mlir::failed(def->fold(operands, results)) || results.empty())
    return {};
  mlir::OpFoldResult folded = results[result.getResultNumber()];
  if (auto foldedValue = mlir::dyn_cast<mlir::Value>(folded))
    return foldedValue == value ? mlir::Attribute()
                                : foldToConstant(foldedValue, depth + 1);
  return mlir::cast<mlir::Attribute>(folded);
}

mlir::Attribute cir::foldToConstant(mlir::Value value) {
  return ::foldToConstant(value, /*depth=*/0);
}

std::optional<bool> cir::tryFoldConstantBool(mlir::Value value) {
  if (auto boolAttr =
          mlir::dyn_cast_if_present<BoolAttr>(foldToConstant(value)))
    return boolAttr.getValue();
  return std::nullopt;
}
