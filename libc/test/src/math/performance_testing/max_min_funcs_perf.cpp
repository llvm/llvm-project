//===-- Performance test for maximum and minimum functions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BinaryOpSingleOutputPerf.h"
#include "src/math/fmaxf.h"
#include "src/math/fmaxf16.h"
#include "src/math/fmaximum_numf.h"
#include "src/math/fmaximum_numf16.h"
#include "src/math/fmaximumf.h"
#include "src/math/fmaximumf16.h"
#include "src/math/fminf.h"
#include "src/math/fminf16.h"
#include "src/math/fminimum_numf.h"
#include "src/math/fminimum_numf16.h"
#include "src/math/fminimumf.h"
#include "src/math/fminimumf16.h"

#include <math.h>

static constexpr size_t FLOAT16_ROUNDS = 20'000;
static constexpr size_t FLOAT_ROUNDS = 40;

// LLVM libc might be the only libc implementation with support for float16 math
// functions currently. We can't compare our float16 functions against the
// system libc, so we compare them against this placeholder function.
float16 placeholder_binaryf16(float16 x, float16 y) { return x; }

// The system libc might not provide the fmaximum* and fminimum* C23 math
// functions either.
float placeholder_binaryf(float x, float y) { return x; }

int main() {
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float16, float16, LIBC_NAMESPACE::fmaxf16,
                                  placeholder_binaryf16, FLOAT16_ROUNDS,
                                  "fmaxf16_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float16, float16, LIBC_NAMESPACE::fminf16,
                                  placeholder_binaryf16, FLOAT16_ROUNDS,
                                  "fminf16_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float16, float16, LIBC_NAMESPACE::fmaximumf16,
                                  placeholder_binaryf16, FLOAT16_ROUNDS,
                                  "fmaximumf16_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float16, float16, LIBC_NAMESPACE::fminimumf16,
                                  placeholder_binaryf16, FLOAT16_ROUNDS,
                                  "fminimumf16_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(
      float16, float16, LIBC_NAMESPACE::fmaximum_numf16, placeholder_binaryf16,
      FLOAT16_ROUNDS, "fmaximum_numf16_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(
      float16, float16, LIBC_NAMESPACE::fminimum_numf16, placeholder_binaryf16,
      FLOAT16_ROUNDS, "fminimum_numf16_perf.log")

  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float, float, LIBC_NAMESPACE::fmaxf, ::fmaxf,
                                  FLOAT_ROUNDS, "fmaxf_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float, float, LIBC_NAMESPACE::fminf, ::fminf,
                                  FLOAT_ROUNDS, "fminf_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float, float, LIBC_NAMESPACE::fmaximumf,
                                  placeholder_binaryf, FLOAT_ROUNDS,
                                  "fmaximumf_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float, float, LIBC_NAMESPACE::fminimumf,
                                  placeholder_binaryf, FLOAT_ROUNDS,
                                  "fminimumf_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float, float, LIBC_NAMESPACE::fmaximum_numf,
                                  placeholder_binaryf, FLOAT_ROUNDS,
                                  "fmaximum_numf_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float, float, LIBC_NAMESPACE::fminimum_numf,
                                  placeholder_binaryf, FLOAT_ROUNDS,
                                  "fminimum_numf_perf.log")

  return 0;
}
