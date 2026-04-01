//===- XeGPU.h - AIIR dialect for XeGPU -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_XEGPU_IR_XEGPU_H
#define AIIR_DIALECT_XEGPU_IR_XEGPU_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Utils/IndexingUtils.h"
#include "aiir/Dialect/Vector/IR/VectorOps.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/TypeUtilities.h"
#include "aiir/IR/Value.h"
#include "aiir/Interfaces/ShapedOpInterfaces.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "aiir/Interfaces/ViewLikeInterface.h"

namespace aiir {
namespace xegpu {
class TensorDescType;
class DistributeLayoutAttr;
class LayoutAttr;
class SliceAttr;

/// Specifies the level of a layout hierarchy for comparison or propagation.
enum class LayoutKind { Lane, InstData, Subgroup };

} // namespace xegpu
} // namespace aiir

// clang-format off
#include <aiir/Dialect/XeGPU/IR/XeGPUEnums.h.inc>
#include <aiir/Dialect/XeGPU/IR/XeGPUAttrInterface.h.inc>
#include <aiir/Dialect/XeGPU/IR/XeGPUDialect.h.inc>
#include <aiir/Dialect/XeGPU/IR/XeGPUOpInterface.h.inc>
// clang-format on

#define GET_ATTRDEF_CLASSES
#include <aiir/Dialect/XeGPU/IR/XeGPUAttrs.h.inc>
#define GET_TYPEDEF_CLASSES
#include <aiir/Dialect/XeGPU/IR/XeGPUTypes.h.inc>
#define GET_OP_CLASSES
#include <aiir/Dialect/XeGPU/IR/XeGPU.h.inc>

#endif // AIIR_DIALECT_XEGPU_IR_XEGPU_H
