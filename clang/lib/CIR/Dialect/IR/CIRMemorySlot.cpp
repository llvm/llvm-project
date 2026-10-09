//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MemorySlot-related interfaces for CIR dialect
// operations.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;

/// Conditions the deletion of the operation to the removal of all its uses.
static bool forwardToUsers(Operation *op,
                           SmallVectorImpl<OpOperand *> &newBlockingUses) {
  for (Value result : op->getResults())
    for (OpOperand &use : result.getUses())
      newBlockingUses.push_back(&use);
  return true;
}

//===----------------------------------------------------------------------===//
// Interfaces for AllocaOp
//===----------------------------------------------------------------------===//

/// Returns true if the scope still contains a cir.goto or cir.indirect_goto.
/// Until GotoSolver turns them into branches, these jumps are terminators
/// without successors, so the block holding the target cir.label does not list
/// them as predecessors.
static bool hasUnresolvedGotos(Operation *scope) {
  return scope
      ->walk([](Block *block) {
        return (!block->empty() &&
                isa<cir::GotoOp, cir::IndirectGotoOp>(block->back()))
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

llvm::SmallVector<MemorySlot> cir::AllocaOp::getPromotableSlots() {
  // Promotion computes reaching definitions from the CFG, which misses the
  // edges of unresolved gotos, so a load after a cir.label would be given the
  // value from the fallthrough path only. Leave the slot in memory until the
  // gotos are lowered.
  if (hasUnresolvedGotos(
          getOperation()->getParentWithTrait<OpTrait::IsIsolatedFromAbove>()))
    return {};
  return {MemorySlot{getResult(), getAllocaType()}};
}

Value cir::AllocaOp::getDefaultValue(const MemorySlot &slot,
                                     OpBuilder &builder) {
  return cir::ConstantOp::create(builder, getLoc(),
                                 cir::UndefAttr::get(slot.valueType));
}

void cir::AllocaOp::handleBlockArgument(const MemorySlot &slot,
                                        BlockArgument argument,
                                        OpBuilder &builder) {}

std::optional<PromotableAllocationOpInterface>
cir::AllocaOp::handlePromotionComplete(const MemorySlot &slot,
                                       Value defaultValue, OpBuilder &builder) {
  if (defaultValue && defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  this->erase();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Interfaces for LoadOp
//===----------------------------------------------------------------------===//

bool cir::LoadOp::loadsFrom(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

bool cir::LoadOp::storesTo(const MemorySlot &slot) { return false; }

Value cir::LoadOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                             Value reachingDef, const DataLayout &dataLayout) {
  llvm_unreachable("getStored should not be called on LoadOp");
}

bool cir::LoadOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;

  // Volatile load or atomic load should not be removed.
  if (getIsVolatile() || getMemOrder().has_value())
    return false;

  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getType() == slot.valueType;
}

DeletionKind cir::LoadOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  getResult().replaceAllUsesWith(reachingDefinition);
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// Interfaces for StoreOp
//===----------------------------------------------------------------------===//

bool cir::StoreOp::loadsFrom(const MemorySlot &slot) { return false; }

bool cir::StoreOp::storesTo(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

Value cir::StoreOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                              Value reachingDef, const DataLayout &dataLayout) {
  return getValue();
}

bool cir::StoreOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;

  // Volatile store or atomic store should not be removed.
  if (getIsVolatile() || getMemOrder().has_value())
    return false;

  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getValue() != slot.ptr && slot.valueType == getValue().getType();
}

DeletionKind cir::StoreOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// Interfaces for CopyOp
//===----------------------------------------------------------------------===//

bool cir::CopyOp::loadsFrom(const MemorySlot &slot) {
  return getSrc() == slot.ptr;
}

bool cir::CopyOp::storesTo(const MemorySlot &slot) {
  return getDst() == slot.ptr;
}

Value cir::CopyOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                             Value reachingDef, const DataLayout &dataLayout) {
  return cir::LoadOp::create(builder, getLoc(), slot.valueType, getSrc());
}

DeletionKind cir::CopyOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, mlir::Value reachingDefinition,
    const DataLayout &dataLayout) {
  if (loadsFrom(slot))
    cir::StoreOp::create(builder, getLoc(), reachingDefinition, getDst(),
                         /*is_volatile=*/false,
                         /*is_nontemporal=*/false,
                         /*alignment=*/mlir::IntegerAttr{},
                         /*sync_scope=*/cir::SyncScopeKindAttr(),
                         /*mem-order=*/cir::MemOrderAttr());
  return DeletionKind::Delete;
}

bool cir::CopyOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (getDst() == getSrc())
    return false;

  return getCopySizeInBytes(dataLayout) ==
         dataLayout.getTypeSize(slot.valueType);
}

//===----------------------------------------------------------------------===//
// Interfaces for MatrixColumnMajorLoadOp
//===----------------------------------------------------------------------===//

bool cir::MatrixColumnMajorLoadOp::loadsFrom(const MemorySlot &slot) {
  return getValue() == slot.ptr;
}

bool cir::MatrixColumnMajorLoadOp::storesTo(const MemorySlot &slot) {
  return false;
}

Value cir::MatrixColumnMajorLoadOp::getStored(const MemorySlot &slot,
                                              OpBuilder &builder,
                                              Value reachingDef,
                                              const DataLayout &dataLayout) {
  llvm_unreachable("getStored should not be called on MatrixColumnMajorLoadOp");
}

bool cir::MatrixColumnMajorLoadOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;

  // Volatile load should not be removed.
  if (getIsVolatile())
    return false;

  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getValue() == slot.ptr &&
         getType() == slot.valueType;
}

DeletionKind cir::MatrixColumnMajorLoadOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  getResult().replaceAllUsesWith(reachingDefinition);
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// Interfaces for MatrixColumnMajorStoreOp
//===----------------------------------------------------------------------===//

bool cir::MatrixColumnMajorStoreOp::loadsFrom(const MemorySlot &slot) {
  return false;
}

bool cir::MatrixColumnMajorStoreOp::storesTo(const MemorySlot &slot) {
  return getValue() == slot.ptr;
}

Value cir::MatrixColumnMajorStoreOp::getStored(const MemorySlot &slot,
                                               OpBuilder &builder,
                                               Value reachingDef,
                                               const DataLayout &dataLayout) {
  return getMatrix();
}

bool cir::MatrixColumnMajorStoreOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;

  // Volatile store should not be removed.
  if (getIsVolatile())
    return false;

  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getValue() == slot.ptr &&
         getValue() != slot.ptr && slot.valueType == getValue().getType();
}

DeletionKind cir::MatrixColumnMajorStoreOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// Interfaces for CastOp
//===----------------------------------------------------------------------===//

bool cir::CastOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (getKind() == cir::CastKind::bitcast)
    return forwardToUsers(*this, newBlockingUses);
  return false;
}

DeletionKind cir::CastOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}
