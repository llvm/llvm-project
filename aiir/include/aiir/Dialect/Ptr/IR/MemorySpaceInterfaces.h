//===-- MemorySpaceInterfaces.h - ptr memory space interfaces ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ptr dialect memory space interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_PTR_IR_MEMORYSPACEINTERFACES_H
#define AIIR_DIALECT_PTR_IR_MEMORYSPACEINTERFACES_H

#include "aiir/IR/Attributes.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/OpDefinition.h"

#include <functional>
#include <optional>

namespace aiir {
class Operation;
class DataLayout;
namespace ptr {
enum class AtomicBinOp : uint32_t;
enum class AtomicOrdering : uint32_t;
} // namespace ptr
} // namespace aiir

#include "aiir/Dialect/Ptr/IR/MemorySpaceAttrInterfaces.h.inc"

#endif // AIIR_DIALECT_PTR_IR_MEMORYSPACEINTERFACES_H
