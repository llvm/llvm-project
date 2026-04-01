//===- PtrDialect.h - Pointer dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Ptr dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_PTR_IR_PTROPS_H
#define AIIR_DIALECT_PTR_IR_PTROPS_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Ptr/IR/PtrAttrs.h"
#include "aiir/Dialect/Ptr/IR/PtrDialect.h"
#include "aiir/Dialect/Ptr/IR/PtrTypes.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "aiir/Interfaces/ViewLikeInterface.h"

#define GET_OP_CLASSES
#include "aiir/Dialect/Ptr/IR/PtrOps.h.inc"

#endif // AIIR_DIALECT_PTR_IR_PTROPS_H
