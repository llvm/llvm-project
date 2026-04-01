//===- PtrTypes.h - Pointer types -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Pointer dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_PTR_IR_PTRTYPES_H
#define AIIR_DIALECT_PTR_IR_PTRTYPES_H

#include "aiir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "aiir/IR/Types.h"
#include "aiir/Interfaces/DataLayoutInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/Ptr/IR/PtrOpsTypes.h.inc"

#endif // AIIR_DIALECT_PTR_IR_PTRTYPES_H
