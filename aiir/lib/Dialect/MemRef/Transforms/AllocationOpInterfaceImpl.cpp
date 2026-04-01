//===- AllocationOpInterfaceImpl.cpp - Impl. of AllocationOpInterface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"

#include "aiir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "aiir/Dialect/Bufferization/IR/Bufferization.h"
#include "aiir/Dialect/MemRef/IR/MemRef.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/Operation.h"

using namespace aiir;

namespace {
struct DefaultAllocationInterface
    : public bufferization::AllocationOpInterface::ExternalModel<
          DefaultAllocationInterface, memref::AllocOp> {
  static std::optional<Operation *> buildDealloc(OpBuilder &builder,
                                                 Value alloc) {
    return memref::DeallocOp::create(builder, alloc.getLoc(), alloc)
        .getOperation();
  }
  static std::optional<Value> buildClone(OpBuilder &builder, Value alloc) {
    return bufferization::CloneOp::create(builder, alloc.getLoc(), alloc)
        .getResult();
  }
  static ::aiir::HoistingKind getHoistingKind() {
    return HoistingKind::Loop | HoistingKind::Block;
  }
  static ::std::optional<::aiir::Operation *>
  buildPromotedAlloc(OpBuilder &builder, Value alloc) {
    Operation *definingOp = alloc.getDefiningOp();
    return memref::AllocaOp::create(
        builder, definingOp->getLoc(),
        cast<MemRefType>(definingOp->getResultTypes()[0]),
        definingOp->getOperands(), definingOp->getAttrs());
  }
};

struct DefaultAutomaticAllocationHoistingInterface
    : public bufferization::AllocationOpInterface::ExternalModel<
          DefaultAutomaticAllocationHoistingInterface, memref::AllocaOp> {
  static ::aiir::HoistingKind getHoistingKind() { return HoistingKind::Loop; }
};

struct DefaultReallocationInterface
    : public bufferization::AllocationOpInterface::ExternalModel<
          DefaultAllocationInterface, memref::ReallocOp> {
  static std::optional<Operation *> buildDealloc(OpBuilder &builder,
                                                 Value realloc) {
    return memref::DeallocOp::create(builder, realloc.getLoc(), realloc)
        .getOperation();
  }
};
} // namespace

void aiir::memref::registerAllocationOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](AIIRContext *ctx, memref::MemRefDialect *dialect) {
    memref::AllocOp::attachInterface<DefaultAllocationInterface>(*ctx);
    memref::AllocaOp::attachInterface<
        DefaultAutomaticAllocationHoistingInterface>(*ctx);
    memref::ReallocOp::attachInterface<DefaultReallocationInterface>(*ctx);
  });
}
