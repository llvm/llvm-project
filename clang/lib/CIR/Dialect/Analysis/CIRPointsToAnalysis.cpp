//===- CIRPointsToAnalysis.cpp - Points-to analysis for CIR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/CIRPointsToAnalysis.h"
#include "mlir/IR/Remarks.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"

#include <algorithm>

#define DEBUG_TYPE "cir-points-to-analysis"

using namespace cir;

/// The name used to print `object`: the name of the cir.alloca allocating it.
static llvm::StringRef getObjectName(mlir::Value object) {
  llvm::StringRef name = object.getDefiningOp<cir::AllocaOp>().getName();
  return name.empty() ? "<unnamed>" : name;
}

/// The options of every remark the analysis emits for `root`.
static mlir::remark::RemarkOpts getRemarkOpts(llvm::StringRef name,
                                              mlir::Operation *root) {
  mlir::remark::RemarkOpts opts =
      mlir::remark::RemarkOpts::name(name).category("points-to");
  if (auto func = mlir::dyn_cast<cir::FuncOp>(root))
    opts = opts.function(func.getSymName());
  return opts;
}

//===----------------------------------------------------------------------===//
// PointsToSet
//===----------------------------------------------------------------------===//

bool PointsToSet::merge(const PointsToSet &other) {
  bool changed = false;
  if (!sizeLimitHit)
    for (mlir::Value object : other.objects)
      changed |= objects.insert(object);
  if (other.unknown)
    changed |= insertUnknown();
  return changed;
}

void PointsToSet::replaceWithUnknown() {
  objects.clear();
  unknown = true;
  sizeLimitHit = true;
}

void PointsToSet::print(llvm::raw_ostream &os) const {
  llvm::SmallVector<llvm::StringRef> names =
      llvm::map_to_vector(objects, getObjectName);
  llvm::sort(names);
  if (unknown)
    names.push_back("unknown");
  os << '[';
  llvm::interleaveComma(names, os);
  os << ']';
}

//===----------------------------------------------------------------------===//
// PointsToAnalysis
//===----------------------------------------------------------------------===//

PointsToAnalysis::PointsToAnalysis(mlir::Operation *op,
                                   const PointsToLimits &limits)
    : limits(limits) {
  unknownSet.insertUnknown();
  compute(op);
  emitRemarks(op);
}

void PointsToAnalysis::compute(mlir::Operation *op) {
  // Create the sets of the values the analysis computes before solving, so a
  // value never changes from "not analyzed", which reads as "unknown", to a set
  // with fewer members. Each cir.alloca result points to the object it
  // allocates, each cir.load result starts out pointing to nothing, and a null
  // pointer constant points to nothing.
  uint64_t numOperations = 0;
  mlir::WalkResult tooLarge = op->walk([&](mlir::Operation *nested) {
    if (nested == op)
      return mlir::WalkResult::advance();
    if (++numOperations > limits.maxOperations)
      return mlir::WalkResult::interrupt();
    if (auto alloca = mlir::dyn_cast<cir::AllocaOp>(nested)) {
      valuePointsTo[alloca.getResult()].insert(alloca.getResult());
    } else if (auto load = mlir::dyn_cast<cir::LoadOp>(nested)) {
      valuePointsTo.try_emplace(load.getResult());
    } else if (auto constant = mlir::dyn_cast<cir::ConstantOp>(nested)) {
      auto ptr = mlir::dyn_cast<cir::ConstPtrAttr>(constant.getValue());
      if (ptr && ptr.isNullValue())
        valuePointsTo.try_emplace(constant.getResult());
    }
    return mlir::WalkResult::advance();
  });
  if (tooLarge.wasInterrupted()) {
    mlir::remark::missed(op->getLoc(), getRemarkOpts("OperationLimit", op))
        << mlir::remark::add("gave up: more than {0} operations",
                             limits.maxOperations);
    dropAllSets();
    return;
  }

  // Sets only grow, so revisiting every operation until nothing changes
  // terminates. The work limit bounds how long that may take.
  uint64_t budget = std::max<uint64_t>(
      limits.minWork, uint64_t(limits.workPerOperation) * numOperations);
  uint64_t work = 0;
  unsigned iterations = 0;
  bool changed;
  do {
    changed = false;
    mlir::WalkResult result = op->walk([&](mlir::Operation *nested) {
      if (nested == op)
        return mlir::WalkResult::advance();
      if (++work > budget)
        return mlir::WalkResult::interrupt();
      changed |= visit(nested);
      return mlir::WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      mlir::remark::missed(op->getLoc(), getRemarkOpts("WorkLimit", op))
          << mlir::remark::add("gave up: more than {0} operation visits",
                               budget);
      dropAllSets();
      return;
    }
    ++iterations;
  } while (changed);
  LDBG() << "Solved in " << iterations << " iterations";
}

void PointsToAnalysis::emitRemarks(mlir::Operation *root) const {
  if (!root->getContext()->getRemarkEngine())
    return;

  for (auto [object, op] : escaped)
    mlir::remark::missed(op->getLoc(), getRemarkOpts("Escaped", root))
        << mlir::remark::add("the address of {0} escapes",
                             getObjectName(object));
  for (auto [op, size] : sizeLimitHits)
    mlir::remark::missed(op->getLoc(), getRemarkOpts("SizeLimit", root))
        << mlir::remark::add(
               "a set of {0} objects is replaced by unknown, the limit is {1}",
               size, limits.maxSetSize);

  root->walk([&](cir::LoadOp load) {
    if (!mlir::isa<cir::PointerType>(load.getType()))
      return;
    std::string set;
    llvm::raw_string_ostream os(set);
    os << getPointsTo(load.getResult());
    mlir::remark::analysis(load.getLoc(), getRemarkOpts("PointsTo", root))
        << mlir::remark::add("points to {0}", set);
  });
}

const PointsToSet &PointsToAnalysis::getPointsTo(mlir::Value value) const {
  auto it = valuePointsTo.find(value);
  if (it == valuePointsTo.end())
    return unknownSet;
  return it->second;
}

void PointsToAnalysis::dropAllSets() {
  // Values without a set point to "unknown", so dropping every set is enough
  // to make every answer conservative.
  valuePointsTo.clear();
  memoryPointsTo.clear();
  escaped.clear();
  sizeLimitHits.clear();
  limitHit = true;
}

bool PointsToAnalysis::addTo(PointsToSet &to, const PointsToSet &from,
                             mlir::Operation *op) {
  // A set that hit the size limit no longer names its objects, so the objects
  // added to it are lost. Make them escape, so that dropping them is
  // conservative.
  if (to.hitSizeLimit())
    return markEscaped(from.getObjects(), op);

  bool changed = to.merge(from);
  if (to.getObjects().size() <= limits.maxSetSize)
    return changed;

  // The objects the set held are lost too. They would escape when the
  // operations that added them are visited again, but making them escape here
  // doesn't depend on the order operations are visited in.
  llvm::SmallVector<mlir::Value> objects(to.getObjects());
  sizeLimitHits.emplace_back(op, objects.size());
  to.replaceWithUnknown();
  // `to` may be an element of memoryPointsTo, which markEscaped() can grow, so
  // it must not be used after this point.
  markEscaped(objects, op);
  return true;
}

bool PointsToAnalysis::markEscaped(llvm::ArrayRef<mlir::Value> objects,
                                   mlir::Operation *op) {
  bool changed = false;
  llvm::SmallVector<mlir::Value> worklist;
  for (mlir::Value object : objects)
    if (escaped.insert({object, op}).second)
      worklist.push_back(object);

  while (!worklist.empty()) {
    mlir::Value object = worklist.pop_back_val();
    changed = true;

    // Code the analysis cannot see may store any address it can reach into an
    // escaped object, and may follow the addresses already stored there.
    PointsToSet &memory = memoryPointsTo[object];
    memory.insertUnknown();
    for (mlir::Value inner : memory.getObjects())
      if (escaped.insert({inner, op}).second)
        worklist.push_back(inner);
  }
  return changed;
}

bool PointsToAnalysis::visit(mlir::Operation *op) {
  if (auto store = mlir::dyn_cast<cir::StoreOp>(op)) {
    const PointsToSet &addr = getPointsTo(store.getAddr());
    const PointsToSet &value = getPointsTo(store.getValue());
    bool changed = false;
    for (mlir::Value object : addr.getObjects()) {
      changed |= addTo(memoryPointsTo[object], value, op);
      if (escaped.contains(object))
        changed |= markEscaped(value.getObjects(), op);
    }
    // A store through an address the analysis does not know may write to an
    // escaped object or to memory it does not track at all.
    if (addr.containsUnknown())
      changed |= markEscaped(value.getObjects(), op);
    return changed;
  }

  if (auto load = mlir::dyn_cast<cir::LoadOp>(op)) {
    const PointsToSet &addr = getPointsTo(load.getAddr());
    PointsToSet loaded;
    for (mlir::Value object : addr.getObjects()) {
      auto it = memoryPointsTo.find(object);
      if (it != memoryPointsTo.end())
        loaded.merge(it->second);
    }
    if (addr.containsUnknown())
      loaded.insertUnknown();
    return addTo(valuePointsTo[load.getResult()], loaded, op);
  }

  // The analysis does not know what any other operation does with an address
  // it receives, so the objects it points to escape. The results of such an
  // operation have no set, and so point to "unknown".
  bool changed = false;
  for (mlir::Value operand : op->getOperands())
    changed |= markEscaped(getPointsTo(operand).getObjects(), op);
  return changed;
}
