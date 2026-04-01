//===- TransformTypes.h - Transform dialect types ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_TRANSFORM_IR_TRANSFORMTYPES_H
#define AIIR_DIALECT_TRANSFORM_IR_TRANSFORMTYPES_H

#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "aiir/IR/Types.h"
#include "aiir/Support/LLVM.h"

namespace aiir {
class DiagnosedSilenceableFailure;
class Operation;
class Type;
} // namespace aiir

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/Transform/IR/TransformTypes.h.inc"

#endif // AIIR_DIALECT_TRANSFORM_IR_TRANSFORMTYPES_H
