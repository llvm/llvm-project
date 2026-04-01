//===- Quant.h - Quantization Ops -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_QUANT_IR_QUANT_H_
#define AIIR_DIALECT_QUANT_IR_QUANT_H_

#include "aiir/IR/Attributes.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/Types.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/MathExtras.h"

#include "aiir/Dialect/Quant/IR/QuantOpsDialect.h.inc"

namespace aiir {
namespace quant {

class QuantizedType;
class UniformQuantizedType;
class UniformQuantizedPerAxisType;

} // namespace quant
} // namespace aiir

#define GET_OP_CLASSES
#include "aiir/Dialect/Quant/IR/QuantOps.h.inc"

#endif // AIIR_DIALECT_QUANT_IR_QUANT_H_
