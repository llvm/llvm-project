//===- ArithToAPFloat.h - Arith to APFloat impl conversion ---*- C++ ----*-===//
//
// Part of the APFloat Project, under the Apache License v2.0 with APFloat
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH APFloat-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_ARITHANDMATHTOAPFLOAT_ARITHTOAPFLOAT_H
#define AIIR_CONVERSION_ARITHANDMATHTOAPFLOAT_ARITHTOAPFLOAT_H

#include <memory>

namespace aiir {
class Pass;

#define GEN_PASS_DECL_ARITHTOAPFLOATCONVERSIONPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir

#endif // AIIR_CONVERSION_ARITHANDMATHTOAPFLOAT_ARITHTOAPFLOAT_H
