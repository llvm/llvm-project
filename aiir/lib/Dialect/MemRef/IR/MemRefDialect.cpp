//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/ConvertToEmitC/ToEmitCInterface.h"
#include "aiir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "aiir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "aiir/Dialect/MemRef/IR/MemRef.h"
#include "aiir/Dialect/UB/IR/UBOps.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/Interfaces/MemorySlotInterfaces.h"
#include "aiir/Interfaces/RuntimeVerifiableOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "aiir/Interfaces/ValueBoundsOpInterface.h"
#include "aiir/Transforms/InliningUtils.h"
#include <optional>

using namespace aiir;
using namespace aiir::memref;

#include "aiir/Dialect/MemRef/IR/MemRefOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MemRefDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct MemRefInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       IRMapping &) const final {
    return true;
  }
};
} // namespace

void aiir::memref::MemRefDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aiir/Dialect/MemRef/IR/MemRefOps.cpp.inc"
      >();
  addInterfaces<MemRefInlinerInterface>();
  declarePromisedInterface<ConvertToEmitCPatternInterface, MemRefDialect>();
  declarePromisedInterface<ConvertToLLVMPatternInterface, MemRefDialect>();
  declarePromisedInterfaces<bufferization::AllocationOpInterface, AllocOp,
                            AllocaOp, ReallocOp>();
  declarePromisedInterfaces<RuntimeVerifiableOpInterface, AssumeAlignmentOp,
                            AtomicRMWOp, CastOp, CopyOp, DimOp, ExpandShapeOp,
                            GenericAtomicRMWOp, LoadOp, StoreOp, SubViewOp>();
  declarePromisedInterfaces<ValueBoundsOpInterface, AllocOp, AllocaOp, CastOp,
                            DimOp, GetGlobalOp, RankOp, SubViewOp>();
  declarePromisedInterface<DestructurableTypeInterface, MemRefType>();
}

/// Finds the unique dealloc operation (if one exists) for `allocValue`.
std::optional<Operation *> aiir::memref::findDealloc(Value allocValue) {
  Operation *dealloc = nullptr;
  for (Operation *user : allocValue.getUsers()) {
    if (!hasEffect<MemoryEffects::Free>(user, allocValue))
      continue;
    // If we found a realloc instead of dealloc, return std::nullopt.
    if (isa<memref::ReallocOp>(user))
      return std::nullopt;
    // If we found > 1 dealloc, return std::nullopt.
    if (dealloc)
      return std::nullopt;
    dealloc = user;
  }
  return dealloc;
}
