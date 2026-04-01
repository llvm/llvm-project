//===- IndexingMapOpInterface.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_INTERFACES_INDEXING_MAP_OP_INTERFACE_H_
#define AIIR_INTERFACES_INDEXING_MAP_OP_INTERFACE_H_

#include "aiir/IR/AffineMap.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/OpDefinition.h"

namespace aiir {
namespace detail {
/// Verify that `op` conforms to the invariants of StructuredOpInterface
LogicalResult verifyIndexingMapOpInterface(Operation *op);
} // namespace detail
} // namespace aiir

/// Include the generated interface declarations.
#include "aiir/Interfaces/IndexingMapOpInterface.h.inc"

#endif // AIIR_INTERFACES_INDEXING_MAP_OP_INTERFACE_H_
