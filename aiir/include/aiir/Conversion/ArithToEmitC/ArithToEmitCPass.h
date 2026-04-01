//===- ArithToEmitCPass.h - Arith to EmitC Pass -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_ARITHTOEMITC_ARITHTOEMITCPASS_H
#define AIIR_CONVERSION_ARITHTOEMITC_ARITHTOEMITCPASS_H

#include <memory>

namespace aiir {
class Pass;

#define GEN_PASS_DECL_CONVERTARITHTOEMITC
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

#endif // AIIR_CONVERSION_ARITHTOEMITC_ARITHTOEMITCPASS_H
