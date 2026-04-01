//===- IndexOps.h - Index operation declarations ------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_INDEX_IR_INDEXOPS_H
#define AIIR_DIALECT_INDEX_IR_INDEXOPS_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Index/IR/IndexAttrs.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/Interfaces/CastInterfaces.h"
#include "aiir/Interfaces/InferIntRangeInterface.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace aiir {
class PatternRewriter;
namespace index {
enum class IndexCmpPredicate : uint32_t;
class IndexCmpPredicateAttr;
} // namespace index
} // namespace aiir

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/Index/IR/IndexOps.h.inc"

#endif // AIIR_DIALECT_INDEX_IR_INDEXOPS_H
