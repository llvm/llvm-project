//===----------------------------------------------------------------------===//
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

#ifndef CLANG_CIR_INTERFACES_CIRLOOPOPINTERFACE_H
#define CLANG_CIR_INTERFACES_CIRLOOPOPINTERFACE_H

#include "llvm/ADT/APInt.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/Operation.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/LoopLikeInterface.h"

using llvm::APInt;
namespace cir {
namespace detail {

/// Verify invariants of the LoopOpInterface.
aiir::LogicalResult verifyLoopOpInterface(::aiir::Operation *op);

} // namespace detail
} // namespace cir

/// Include the tablegen'd interface declarations.
#include "clang/CIR/Interfaces/CIRLoopOpInterface.h.inc"

#endif // CLANG_CIR_INTERFACES_CIRLOOPOPINTERFACE_H
