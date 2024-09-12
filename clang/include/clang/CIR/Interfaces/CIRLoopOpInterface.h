//===- CIRLoopOpInterface.h - Interface for CIR loop-like ops --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// Defines the interface to generically handle CIR loop operations.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_INTERFACES_CIR_CIRLOOPOPINTERFACE_H_
#define CLANG_INTERFACES_CIR_CIRLOOPOPINTERFACE_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir {
namespace cir {
namespace detail {

/// Verify invariants of the LoopOpInterface.
::mlir::LogicalResult verifyLoopOpInterface(::mlir::Operation *op);

} // namespace detail
} // namespace cir
} // namespace mlir

/// Include the tablegen'd interface declarations.
#include "clang/CIR/Interfaces/CIRLoopOpInterface.h.inc"

#endif // CLANG_INTERFACES_CIR_CIRLOOPOPINTERFACE_H_
