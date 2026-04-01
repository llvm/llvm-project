//===- MemoryAccessOpInterfaces.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_MEMREF_IR_MEMORYACCESSOPINTERFACES_H
#define AIIR_DIALECT_MEMREF_IR_MEMORYACCESSOPINTERFACES_H

#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/Operation.h"
#include "aiir/Support/LLVM.h"

namespace aiir {
class RewriterBase;

namespace memref::detail {
LogicalResult verifyIndexedAccessOpInterface(Operation *op);
LogicalResult verifyIndexedMemCopyOpInterface(Operation *op);
} // namespace memref::detail
} // namespace aiir

//===----------------------------------------------------------------------===//
// Memory Access Op Interfaces
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h.inc"

#endif // AIIR_DIALECT_MEMREF_IR_MEMORYACCESSOPINTERFACES_H
