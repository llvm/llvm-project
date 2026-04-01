//===- Dialect.cpp - Implementation of the linalg dialect and types -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg dialect types and dialect.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "aiir/Dialect/Linalg/IR/Linalg.h"
#include "aiir/Dialect/Math/IR/Math.h"
#include "aiir/Dialect/MemRef/IR/MemRef.h"
#include "aiir/Dialect/Shard/Interfaces/ShardingInterface.h"
#include "aiir/Dialect/Tensor/IR/Tensor.h"
#include "aiir/IR/DialectImplementation.h"
#include "aiir/Interfaces/SubsetOpInterface.h"
#include "aiir/Interfaces/ValueBoundsOpInterface.h"
#include "aiir/Support/LLVM.h"
#include "aiir/Transforms/InliningUtils.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace aiir;
using namespace aiir::linalg;

//===----------------------------------------------------------------------===//
// LinalgDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct LinalgInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  // Operations in Linalg dialect are always legal to inline.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {}
};

} // namespace

//===----------------------------------------------------------------------===//
// LinalgDialect
//===----------------------------------------------------------------------===//

/// Attribute name used to memoize indexing maps for named ops.
constexpr const ::llvm::StringLiteral
    LinalgDialect::kMemoizedIndexingMapsAttrName;

/// Trait to check if T provides a `regionBuilder` method.
template <typename T, typename... Args>
using has_region_builder = decltype(T::regionBuilder);
template <typename T>
using detect_has_region_builder = llvm::is_detected<has_region_builder, T>;

/// SFINAE helper for single C++ class without a `regionBuilder` method (e.g.
/// an OpInterface).
template <typename OpType, typename = std::enable_if_t<
                               !detect_has_region_builder<OpType>::value>>
static void addNamedOpBuilderImpl(
    llvm::StringMap<LinalgDialect::RegionBuilderFunType> &map) {
  // Do nothing.
}

template <typename OpType,
          typename = std::enable_if_t<detect_has_region_builder<OpType>::value>,
          typename = void>
static void addNamedOpBuilderImpl(
    llvm::StringMap<LinalgDialect::RegionBuilderFunType> &map) {
  map.insert(std::make_pair(
      OpType::getOperationName(),
      static_cast<LinalgDialect::RegionBuilderFunType>(OpType::regionBuilder)));
}

template <typename... OpTypes>
static void
addNamedOpBuilders(llvm::StringMap<LinalgDialect::RegionBuilderFunType> &map) {
  (addNamedOpBuilderImpl<OpTypes>(map), ...);
}

void aiir::linalg::LinalgDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aiir/Dialect/Linalg/IR/LinalgOpsAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "aiir/Dialect/Linalg/IR/LinalgOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "aiir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "aiir/Dialect/Linalg/IR/LinalgRelayoutOps.cpp.inc"
      >();

  // Fill the Linalg-specific OpName to RegionBuilder map.
  addNamedOpBuilders<
#define GET_OP_LIST
#include "aiir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >(namedStructuredOpRegionBuilders);

  addInterfaces<LinalgInlinerInterface>();

  declarePromisedInterface<shard::ShardingInterface, GenericOp>();
  declarePromisedInterfaces<shard::ShardingInterface,
#define GET_OP_LIST
#include "aiir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                            >();
  declarePromisedInterface<SubsetOpInterface, CopyOp>();
  declarePromisedInterface<SubsetInsertionOpInterface, CopyOp>();

  // ValueBoundsOpInterface
  declarePromisedInterface<ValueBoundsOpInterface, IndexOp>();

  declarePromisedInterface<PartialReductionOpInterface, linalg::GenericOp>();

  // Tiling Interface
  declarePromisedInterface<TilingInterface, linalg::GenericOp>();
  declarePromisedInterfaces<TilingInterface,
#define GET_OP_LIST
#include "aiir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                            >();
  declarePromisedInterfaces<TilingInterface,
#define GET_OP_LIST
#include "aiir/Dialect/Linalg/IR/LinalgRelayoutOps.cpp.inc"
                            >();
  declarePromisedInterfaces<PartialReductionOpInterface,
#define GET_OP_LIST
#include "aiir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                            >();
  declarePromisedInterfaces<bufferization::BufferizableOpInterface,
#define GET_OP_LIST
#include "aiir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                            >();
}

LogicalResult LinalgDialect::verifyOperationAttribute(Operation *op,
                                                      NamedAttribute attr) {
  if (attr.getName() == LinalgDialect::kMemoizedIndexingMapsAttrName)
    return success();
  return op->emitError() << "attribute '" << attr.getName()
                         << "' not supported by the linalg dialect";
}

#include "aiir/Dialect/Linalg/IR/LinalgOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/Linalg/IR/LinalgOpsAttrDefs.cpp.inc"

#include "aiir/Dialect/Linalg/IR/LinalgOpsDialect.cpp.inc"
