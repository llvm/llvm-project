//===- LoopDistribution.cpp - Distribute reduction loops -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Distribute buried reductions and interchange the exposed loop nest.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_LOOPDISTRIBUTION
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct CountedLoop {
  cir::ForOp op;
  mlir::Value ivSlot;
};

// Target CIRGen's raw form, where IVs load from allocas and addresses compare
// by SSA identity.
static std::optional<CountedLoop> recognizeCountedLoop(cir::ForOp loop) {
  if (!loop.getCond().hasOneBlock() || !loop.getStep().hasOneBlock())
    return std::nullopt;
  auto condition = mlir::dyn_cast_or_null<cir::ConditionOp>(
      loop.getCond().front().getTerminator());
  if (!condition)
    return std::nullopt;
  auto cmp = condition.getCondition().getDefiningOp<cir::CmpOp>();
  if (!cmp || cmp.getKind() != cir::CmpOpKind::lt)
    return std::nullopt;
  auto ivLoad = cmp.getLhs().getDefiningOp<cir::LoadOp>();
  if (!ivLoad || !ivLoad.getAddr().getDefiningOp<cir::AllocaOp>())
    return std::nullopt;

  mlir::Value ivSlot = ivLoad.getAddr();
  bool stepWritesIV = false;
  for (auto store : loop.getStep().front().getOps<cir::StoreOp>())
    stepWritesIV |= store.getAddr() == ivSlot;
  if (!stepWritesIV)
    return std::nullopt;

  bool bodyWritesIV = false;
  loop.getBody().walk(
      [&](cir::StoreOp store) { bodyWritesIV |= store.getAddr() == ivSlot; });
  if (bodyWritesIV)
    return std::nullopt;
  return CountedLoop{loop, ivSlot};
}

static void peelAddress(mlir::Value address,
                        llvm::SmallVectorImpl<mlir::Value> &indices) {
  while (true) {
    if (auto element = address.getDefiningOp<cir::GetElementOp>()) {
      indices.push_back(element.getIndex());
      address = element.getBase();
      continue;
    }
    if (auto stride = address.getDefiningOp<cir::PtrStrideOp>()) {
      indices.push_back(stride.getStride());
      address = stride.getBase();
      continue;
    }
    return;
  }
}

template <typename Predicate>
static bool valueContainsLoad(mlir::Value value, Predicate predicate) {
  llvm::SmallVector<mlir::Value, 8> worklist{value};
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (!worklist.empty()) {
    mlir::Operation *def = worklist.pop_back_val().getDefiningOp();
    if (!def || !visited.insert(def).second)
      continue;
    if (auto load = mlir::dyn_cast<cir::LoadOp>(def))
      if (predicate(load))
        return true;
    llvm::append_range(worklist, def->getOperands());
  }
  return false;
}

// Checks whether a backward slice loads an IV slot or accumulator address.
static bool valueLoadsFrom(mlir::Value value, mlir::Value address) {
  return valueContainsLoad(
      value, [&](cir::LoadOp load) { return load.getAddr() == address; });
}

static bool containsLoop(mlir::Operation *op) {
  bool found = false;
  op->walk([&](cir::ForOp) { found = true; });
  return found;
}

static std::optional<CountedLoop> recognizeLoopScope(cir::ScopeOp scope) {
  cir::ForOp loop;
  for (mlir::Operation &op : scope.getScopeRegion().front())
    if (auto candidate = mlir::dyn_cast<cir::ForOp>(op)) {
      if (loop)
        return std::nullopt;
      loop = candidate;
    }
  if (!loop)
    return std::nullopt;
  return recognizeCountedLoop(loop);
}

struct DistributionCandidate {
  CountedLoop outer;
  CountedLoop reduction;
};

// Structural matcher for the reduction idiom.
static std::optional<DistributionCandidate>
recognizeCandidate(cir::ScopeOp outerScope) {
  std::optional<CountedLoop> outer = recognizeLoopScope(outerScope);
  if (!outer)
    return std::nullopt;

  cir::ScopeOp bodyScope;
  for (mlir::Operation &op : outer->op.getBody().front()) {
    if (mlir::isa<cir::YieldOp>(op))
      continue;
    auto scope = mlir::dyn_cast<cir::ScopeOp>(op);
    if (!scope || bodyScope)
      return std::nullopt;
    bodyScope = scope;
  }
  if (!bodyScope)
    return std::nullopt;

  cir::ScopeOp reductionScope;
  bool hasSurroundingStatement = false;
  for (mlir::Operation &op : bodyScope.getScopeRegion().front()) {
    auto scope = mlir::dyn_cast<cir::ScopeOp>(op);
    if (scope && containsLoop(scope)) {
      if (reductionScope)
        return std::nullopt;
      reductionScope = scope;
    } else if (mlir::isa<cir::ForOp>(op)) {
      return std::nullopt;
    } else if (!mlir::isa<cir::YieldOp>(op)) {
      hasSurroundingStatement = true;
    }
  }
  if (!reductionScope || !hasSurroundingStatement)
    return std::nullopt;

  std::optional<CountedLoop> reduction = recognizeLoopScope(reductionScope);
  if (!reduction)
    return std::nullopt;
  bool hasNestedLoop = false;
  reduction->op.getBody().walk(
      [&](cir::ForOp loop) { hasNestedLoop |= loop != reduction->op; });
  if (hasNestedLoop)
    return std::nullopt;

  cir::StoreOp accumulatorStore;
  bool multipleStores = false;
  reduction->op.getBody().walk([&](cir::StoreOp store) {
    llvm::SmallVector<mlir::Value, 4> indices;
    peelAddress(store.getAddr(), indices);
    if (indices.empty() || llvm::any_of(indices, [&](mlir::Value index) {
          return valueLoadsFrom(index, reduction->ivSlot);
        }))
      return;
    if (accumulatorStore)
      multipleStores = true;
    else
      accumulatorStore = store;
  });
  // Compound assignments reuse one address SSA value; otherwise this misses.
  if (!accumulatorStore || multipleStores ||
      !valueLoadsFrom(accumulatorStore.getValue(), accumulatorStore.getAddr()))
    return std::nullopt;

  return DistributionCandidate{*outer, *reduction};
}

static bool isProfitable(DistributionCandidate candidate) {
  bool profitable = false;
  candidate.reduction.op.getBody().walk([&](cir::LoadOp load) {
    llvm::SmallVector<mlir::Value, 4> indices;
    peelAddress(load.getAddr(), indices);
    if (indices.size() < 2)
      return;
    bool variesWithReduction = llvm::any_of(indices, [&](mlir::Value index) {
      return valueLoadsFrom(index, candidate.reduction.ivSlot);
    });
    // indices.front() is the innermost subscript, the one the interchange
    // makes contiguous.
    if (variesWithReduction &&
        valueLoadsFrom(indices.front(), candidate.outer.ivSlot))
      profitable = true;
  });
  return profitable;
}

struct LoopDistributionPass
    : public impl::LoopDistributionBase<LoopDistributionPass> {
  using impl::LoopDistributionBase<LoopDistributionPass>::LoopDistributionBase;

  void runOnOperation() override {
    // Report each recognized candidate.
    getOperation()->walk([&](cir::FuncOp function) {
      function.walk([&](cir::ScopeOp scope) {
        std::optional<DistributionCandidate> candidate =
            recognizeCandidate(scope);
        if (!candidate || !isProfitable(*candidate))
          return;
        ++numCandidates;
        scope.emitRemark() << "loop distribution candidate in function '"
                           << function.getSymName() << "'";
      });
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createLoopDistributionPass() {
  return std::make_unique<LoopDistributionPass>();
}
