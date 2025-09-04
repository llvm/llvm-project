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

llvm::SmallVector<MemorySlot> cir::AllocaOp::getPromotableSlots() {
  return {MemorySlot{getResult(), getAllocaType()}};
}

Value cir::AllocaOp::getDefaultValue(const MemorySlot &slot,
                                     OpBuilder &builder) {
  return builder.create<cir::ConstantOp>(getLoc(),
                                         cir::UndefAttr::get(slot.elemType));
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
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getType() == slot.elemType;
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
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getValue() != slot.ptr && slot.elemType == getValue().getType();
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
  return cir::LoadOp::create(builder, getLoc(), slot.elemType, getSrc());
}

DeletionKind cir::CopyOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, mlir::Value reachingDefinition,
    const DataLayout &dataLayout) {
  if (loadsFrom(slot))
    cir::StoreOp::create(builder, getLoc(), reachingDefinition, getDst(),
                         /*isVolatile=*/false,
                         /*alignment=*/mlir::IntegerAttr{},
                         /*mem-order=*/cir::MemOrderAttr());
  return DeletionKind::Delete;
}

bool cir::CopyOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (getDst() == getSrc())
    return false;

  return getLength(dataLayout) == dataLayout.getTypeSize(slot.elemType);
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
