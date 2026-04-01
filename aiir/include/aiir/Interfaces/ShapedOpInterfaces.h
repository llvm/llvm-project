//===- ShapedOpInterfaces.h - Interfaces for Shaped Ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces for ops that operate on shaped values.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_INTERFACES_SHAPEDOPINTERFACES_H_
#define AIIR_INTERFACES_SHAPEDOPINTERFACES_H_

#include "aiir/IR/OpDefinition.h"

namespace aiir {
namespace detail {

/// Verify invariants of ops that implement the ShapedDimOpInterface.
LogicalResult verifyShapedDimOpInterface(Operation *op);

} // namespace detail
} // namespace aiir

/// Include the generated interface declarations.
#include "aiir/Interfaces/ShapedOpInterfaces.h.inc"

#endif // AIIR_INTERFACES_SHAPEDOPINTERFACES_H_
