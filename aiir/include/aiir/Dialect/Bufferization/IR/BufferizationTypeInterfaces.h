//===- BufferizationTypeInterfaces.h - Type Interfaces ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATIONTYPEINTERFACES_H_
#define AIIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATIONTYPEINTERFACES_H_

//===----------------------------------------------------------------------===//
// Bufferization Type Interfaces
//===----------------------------------------------------------------------===//

#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/Types.h"

namespace aiir::bufferization {
struct BufferizationOptions;
class BufferLikeType;
} // namespace aiir::bufferization

#include "aiir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.h.inc"

#endif // AIIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATIONTYPEINTERFACES_H_
