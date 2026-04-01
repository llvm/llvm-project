//===- IRDL.h - IR Definition Language dialect ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the dialect for the IR Definition Language.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_IRDL_IR_IRDL_H_
#define AIIR_DIALECT_IRDL_IR_IRDL_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/IRDL/IR/IRDLInterfaces.h"
#include "aiir/Dialect/IRDL/IR/IRDLTraits.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

#include <memory>

// Forward declaration.
namespace aiir {
namespace irdl {
class OpDef;
class OpDefAttr;
} // namespace irdl
} // namespace aiir

//===----------------------------------------------------------------------===//
// IRDL Dialect
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/IRDL/IR/IRDLDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/IRDL/IR/IRDLTypesGen.h.inc"

#include "aiir/Dialect/IRDL/IR/IRDLEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/IRDL/IR/IRDLAttributes.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/IRDL/IR/IRDLOps.h.inc"

#endif // AIIR_DIALECT_IRDL_IR_IRDL_H_
