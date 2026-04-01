//===- MemRef.h - MemRef dialect --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_MEMREF_IR_MEMREF_H_
#define AIIR_DIALECT_MEMREF_IR_MEMREF_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Utils/ReshapeOpsUtils.h"
#include "aiir/IR/Dialect.h"
#include "aiir/Interfaces/AlignmentAttrInterface.h"
#include "aiir/Interfaces/CallInterfaces.h"
#include "aiir/Interfaces/CastInterfaces.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/InferIntRangeInterface.h"
#include "aiir/Interfaces/InferStridedMetadataInterface.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/MemOpInterfaces.h"
#include "aiir/Interfaces/MemorySlotInterfaces.h"
#include "aiir/Interfaces/ShapedOpInterfaces.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "aiir/Interfaces/ViewLikeInterface.h"

#include <optional>

namespace aiir {

namespace arith {
enum class AtomicRMWKind : uint64_t;
class AtomicRMWKindAttr;
} // namespace arith

class Location;
class OpBuilder;

raw_ostream &operator<<(raw_ostream &os, const Range &range);

/// Return the list of Range (i.e. offset, size, stride). Each Range
/// entry contains either the dynamic value or a ConstantIndexOp constructed
/// with `b` at location `loc`.
SmallVector<Range, 8> getOrCreateRanges(OffsetSizeAndStrideOpInterface op,
                                        OpBuilder &b, Location loc);

namespace memref {

/// This is a common utility used for patterns of the form
/// "someop(memref.cast) -> someop". It folds the source of any memref.cast
/// into the root operation directly.
LogicalResult foldMemRefCast(Operation *op, Value inner = nullptr);

/// Return an unranked/ranked tensor type for the given unranked/ranked memref
/// type.
Type getTensorTypeFromMemRefType(Type type);

/// Finds a single dealloc operation for the given allocated value. If there
/// are > 1 deallocates for `allocValue`, returns std::nullopt, else returns the
/// single deallocate if it exists or nullptr.
std::optional<Operation *> findDealloc(Value allocValue);

/// Return the dimension of the given memref value.
OpFoldResult getMixedSize(OpBuilder &builder, Location loc, Value value,
                          int64_t dim);

/// Return the dimensions of the given memref value.
SmallVector<OpFoldResult> getMixedSizes(OpBuilder &builder, Location loc,
                                        Value value);

/// Create a rank-reducing SubViewOp @[0 .. 0] with strides [1 .. 1] and
/// appropriate sizes (i.e. `memref.getSizes()`) to reduce the rank of `memref`
/// to that of `targetShape`.
Value createCanonicalRankReducingSubViewOp(OpBuilder &b, Location loc,
                                           Value memref,
                                           ArrayRef<int64_t> targetShape);
} // namespace memref
} // namespace aiir

//===----------------------------------------------------------------------===//
// MemRef Dialect
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/MemRef/IR/MemRefOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// MemRef Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/MemRef/IR/MemRefOps.h.inc"

#endif // AIIR_DIALECT_MEMREF_IR_MEMREF_H_
