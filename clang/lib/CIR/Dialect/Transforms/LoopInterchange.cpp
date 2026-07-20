//===- LoopInterchange.cpp - recognize loop interchange candidates --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Recognizes loop interchange candidates. See the pass description in
// Passes.td.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_LOOPINTERCHANGE
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct CountedLoop {
  mlir::Value ivSlot;
  mlir::Value bound;
};

static bool storesToSlot(mlir::Region &region, mlir::Value slot) {
  bool found = false;
  region.walk([&](cir::StoreOp store) {
    if (store.getAddr() != slot)
      return mlir::WalkResult::advance();
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

static std::optional<CountedLoop> recognizeCountedLoop(cir::ForOp forOp) {
  mlir::Region &cond = forOp.getCond();
  if (!cond.hasOneBlock())
    return std::nullopt;
  auto condOp =
      mlir::dyn_cast_or_null<cir::ConditionOp>(cond.front().getTerminator());
  if (!condOp)
    return std::nullopt;
  auto cmp = condOp.getCondition().getDefiningOp<cir::CmpOp>();
  // Only the iv < bound form is matched. le and ne are left for a later patch.
  if (!cmp || cmp.getKind() != cir::CmpOpKind::lt)
    return std::nullopt;
  auto ivLoad = cmp.getLhs().getDefiningOp<cir::LoadOp>();
  if (!ivLoad)
    return std::nullopt;
  mlir::Value ivSlot = ivLoad.getAddr();
  if (!ivSlot.getDefiningOp<cir::AllocaOp>())
    return std::nullopt;
  // The step must write the iv. A later patch checks it is a unit increment.
  if (!storesToSlot(forOp.getStep(), ivSlot) ||
      storesToSlot(forOp.getBody(), ivSlot))
    return std::nullopt;
  return CountedLoop{ivSlot, cmp.getRhs()};
}

static void peelAddress(mlir::Value addr,
                        llvm::SmallVectorImpl<mlir::Value> &indices) {
  while (true) {
    if (auto ge = addr.getDefiningOp<cir::GetElementOp>()) {
      indices.push_back(ge.getIndex());
      addr = ge.getBase();
      continue;
    }
    if (auto ps = addr.getDefiningOp<cir::PtrStrideOp>()) {
      indices.push_back(ps.getStride());
      addr = ps.getBase();
      continue;
    }
    break;
  }
}

static bool loadsFromSlot(mlir::Value v, mlir::Value slot) {
  while (auto cast = v.getDefiningOp<cir::CastOp>())
    v = cast.getSrc();
  auto load = v.getDefiningOp<cir::LoadOp>();
  return load && load.getAddr() == slot;
}

static bool dependsOnSlot(mlir::Value v, mlir::Value slot) {
  llvm::SmallVector<mlir::Value> work{v};
  llvm::SmallPtrSet<mlir::Value, 8> seen;
  while (!work.empty()) {
    mlir::Value cur = work.pop_back_val();
    if (!seen.insert(cur).second)
      continue;
    if (auto load = cur.getDefiningOp<cir::LoadOp>()) {
      if (load.getAddr() == slot)
        return true;
      continue;
    }
    if (mlir::Operation *def = cur.getDefiningOp())
      work.append(def->operand_begin(), def->operand_end());
  }
  return false;
}

static cir::ForOp singleInnerFor(cir::ForOp outer) {
  llvm::SmallVector<cir::ForOp> inner;
  outer.getBody().walk([&](cir::ForOp f) {
    if (f != outer && f->getParentOfType<cir::ForOp>() == outer)
      inner.push_back(f);
  });
  return inner.size() == 1 ? inner.front() : cir::ForOp{};
}

// A perfect nest holds nothing but the inner loop and its iv setup in the
// outer body. A stray load, store, or call there makes it imperfect.
static bool isPerfectNest(cir::ForOp outer, cir::ForOp inner,
                          mlir::Value innerSlot) {
  bool perfect = true;
  outer.getBody().walk([&](mlir::Operation *op) {
    if (inner->isAncestor(op))
      return;
    if (auto store = mlir::dyn_cast<cir::StoreOp>(op))
      perfect &= store.getAddr() == innerSlot;
    else if (mlir::isa<cir::LoadOp, cir::CallOp>(op))
      perfect = false;
  });
  return perfect;
}

// True when an access uses the outer iv for the contiguous dimension and the
// inner iv for the next one out, so the inner loop strides down a column.
static bool hasColumnStridedAccess(cir::ForOp inner, mlir::Value outerSlot,
                                   mlir::Value innerSlot) {
  bool found = false;
  inner.getBody().walk([&](mlir::Operation *op) {
    mlir::Value addr;
    if (auto load = mlir::dyn_cast<cir::LoadOp>(op))
      addr = load.getAddr();
    else if (auto store = mlir::dyn_cast<cir::StoreOp>(op))
      addr = store.getAddr();
    else
      return mlir::WalkResult::advance();
    llvm::SmallVector<mlir::Value> idx;
    peelAddress(addr, idx);
    if (idx.size() != 2 || !loadsFromSlot(idx[0], outerSlot) ||
        !loadsFromSlot(idx[1], innerSlot))
      return mlir::WalkResult::advance();
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

// The one recognition entry point. Future shapes are added here, or by
// relaxing one of the predicates it calls.
static bool isInterchangeCandidate(cir::ForOp outer) {
  std::optional<CountedLoop> o = recognizeCountedLoop(outer);
  if (!o)
    return false;
  cir::ForOp inner = singleInnerFor(outer);
  if (!inner)
    return false;
  std::optional<CountedLoop> i = recognizeCountedLoop(inner);
  // Skip a triangular nest whose inner bound varies with the outer iv.
  return i && isPerfectNest(outer, inner, i->ivSlot) &&
         !dependsOnSlot(i->bound, o->ivSlot) &&
         hasColumnStridedAccess(inner, o->ivSlot, i->ivSlot);
}

struct LoopInterchangePass
    : public impl::LoopInterchangeBase<LoopInterchangePass> {
  using impl::LoopInterchangeBase<LoopInterchangePass>::LoopInterchangeBase;

  void runOnOperation() override {
    getOperation()->walk([&](cir::ForOp outer) {
      if (!isInterchangeCandidate(outer))
        return;
      ++numCandidates;
      auto func = outer->getParentOfType<cir::FuncOp>();
      assert(func && "candidate loop outside a function");
      outer.emitRemark() << "loop interchange candidate in function '"
                         << func.getSymName() << "'";
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createLoopInterchangePass() {
  return std::make_unique<LoopInterchangePass>();
}
