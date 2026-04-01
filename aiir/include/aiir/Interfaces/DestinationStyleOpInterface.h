//===- DestinationStyleOpInterface.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_INTERFACES_DESTINATIONSTYLEOPINTERFACE_H_
#define AIIR_INTERFACES_DESTINATIONSTYLEOPINTERFACE_H_

#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/IRMapping.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace aiir {
namespace detail {
/// Verify that `op` conforms to the invariants of DestinationStyleOpInterface
LogicalResult verifyDestinationStyleOpInterface(Operation *op);
} // namespace detail
} // namespace aiir

/// Include the generated interface declarations.
#include "aiir/Interfaces/DestinationStyleOpInterface.h.inc"

#endif // AIIR_INTERFACES_DESTINATIONSTYLEOPINTERFACE_H_
