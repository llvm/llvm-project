//===-- Performance test for maximum and minimum functions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BinaryOpSingleOutputPerf.h"
#include "src/math/fmul.h"
#include "src/math/fmull.h"
#include <math.h>

static constexpr size_t DOUBLE_ROUNDS = 40;
static constexpr size_t LONG_DOUBLE_ROUNDS = 40;

float fmul_placeholder_binary(double x, double y) {
  return static_cast<float>(x*y);
}

float fmull_placeholder_binary(long double x, long double y) {
  return static_cast<float>(x*y);
}

int main() {
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float, double, LIBC_NAMESPACE::fmul,
				  fmul_placeholder_binary, DOUBLE_ROUNDS,
				  "fmul_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float, long double, LIBC_NAMESPACE::fmull,
				  fmull_placeholder_binary, LONG_DOUBLE_ROUNDS,
				  "fmull_perf.log")
  return 0;
}
    
