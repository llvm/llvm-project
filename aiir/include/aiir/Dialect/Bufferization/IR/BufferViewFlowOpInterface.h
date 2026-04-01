//===- BufferViewFlowOpInterface.h - Buffer View Flow Analysis --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_BUFFERIZATION_IR_BUFFERVIEWFLOWOPINTERFACE_H_
#define AIIR_DIALECT_BUFFERIZATION_IR_BUFFERVIEWFLOWOPINTERFACE_H_

#include "aiir/IR/OpDefinition.h"
#include "aiir/Support/LLVM.h"

namespace aiir {
class ValueRange;

namespace bufferization {

using RegisterDependenciesFn = std::function<void(ValueRange, ValueRange)>;

} // namespace bufferization
} // namespace aiir

#include "aiir/Dialect/Bufferization/IR/BufferViewFlowOpInterface.h.inc"

#endif // AIIR_DIALECT_BUFFERIZATION_IR_BUFFERVIEWFLOWOPINTERFACE_H_
