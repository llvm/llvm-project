//===- PtrAttrs.h - Pointer dialect attributes ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Ptr dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_PTR_IR_PTRATTRS_H
#define AIIR_DIALECT_PTR_IR_PTRATTRS_H

#include "aiir/IR/BuiltinAttributeInterfaces.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/Support/TypeSize.h"

#include "aiir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "aiir/Dialect/Ptr/IR/PtrEnums.h"

namespace aiir {
namespace ptr {
class PtrType;
} // namespace ptr
} // namespace aiir

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/Ptr/IR/PtrOpsAttrs.h.inc"

#endif // AIIR_DIALECT_PTR_IR_PTRATTRS_H
