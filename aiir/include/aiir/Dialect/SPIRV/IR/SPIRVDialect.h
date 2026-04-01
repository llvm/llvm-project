//===- SPIRVDialect.h - AIIR SPIR-V dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SPIR-V dialect in AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SPIRV_IR_SPIRVDIALECT_H_
#define AIIR_DIALECT_SPIRV_IR_SPIRVDIALECT_H_

#include "aiir/IR/Dialect.h"

namespace aiir {
namespace spirv {

enum class Decoration : uint32_t;

} // namespace spirv
} // namespace aiir

#include "aiir/Dialect/SPIRV/IR/SPIRVOpsDialect.h.inc"

#endif // AIIR_DIALECT_SPIRV_IR_SPIRVDIALECT_H_
