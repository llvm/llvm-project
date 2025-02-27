//===-- Performance test for miscellaneous basic operations ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BinaryOpSingleOutputPerf.h"
#include "SingleInputSingleOutputPerf.h"
#include "src/math/copysignf.h"
#include "src/math/copysignf16.h"
#include "src/math/fabsf.h"
#include "src/math/fabsf16.h"

#include <math.h>

static constexpr size_t FLOAT16_ROUNDS = 20'000;
static constexpr size_t FLOAT_ROUNDS = 40;

// LLVM libc might be the only libc implementation with support for float16 math
// functions currently. We can't compare our float16 functions against the
// system libc, so we compare them against this placeholder function.
float16 placeholder_unaryf16(float16 x) { return x; }
float16 placeholder_binaryf16(float16 x, float16 y) { return x; }

int main() {
  SINGLE_INPUT_SINGLE_OUTPUT_PERF_EX(float16, LIBC_NAMESPACE::fabsf16,
                                     placeholder_unaryf16, FLOAT16_ROUNDS,
                                     "fabsf16_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float16, float16, LIBC_NAMESPACE::copysignf16,
                                  placeholder_binaryf16, FLOAT16_ROUNDS,
                                  "copysignf16_perf.log")

  SINGLE_INPUT_SINGLE_OUTPUT_PERF_EX(float, LIBC_NAMESPACE::fabsf, fabsf,
                                     FLOAT_ROUNDS, "fabsf_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float, float, LIBC_NAMESPACE::copysignf,
                                  copysignf, FLOAT_ROUNDS, "copysignf_perf.log")

  return 0;
}
