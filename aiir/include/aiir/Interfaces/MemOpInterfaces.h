//===- MemOpInterfaces.h - Memory operation interfaces ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of interfaces for operations that interact
// with memory.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_INTERFACES_MEMOPINTERFACES_H
#define AIIR_INTERFACES_MEMOPINTERFACES_H

#include "aiir/IR/OpDefinition.h"

namespace aiir {
namespace detail {
/// Attempt to verify the given memory space cast operation.
LogicalResult verifyMemorySpaceCastOpInterface(Operation *op);

/// Tries to bubble-down inplace a `MemorySpaceCastOpInterface` operation
/// referenced by `operand`. On success, it returns `std::nullopt`. It
/// returns failure if `operand` doesn't reference a
/// `MemorySpaceCastOpInterface` op.
FailureOr<std::optional<SmallVector<Value>>>
bubbleDownInPlaceMemorySpaceCastImpl(OpOperand &operand, ValueRange results);
} // namespace detail
} // namespace aiir

/// Include the generated interface declarations.
#include "aiir/Interfaces/MemOpInterfaces.h.inc"

#endif // AIIR_INTERFACES_MEMOPINTERFACES_H
