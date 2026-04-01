//===- MemRefToEmitCPass.h - A Pass to convert MemRef to EmitC ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITCPASS_H
#define AIIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITCPASS_H

#include <memory>

namespace aiir {
class Pass;

#define GEN_PASS_DECL_CONVERTMEMREFTOEMITC
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

#endif // AIIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITCPASS_H
