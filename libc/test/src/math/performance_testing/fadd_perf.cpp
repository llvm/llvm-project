//===-- Performance test for the fadd function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BinaryOpSingleOutputPerf.h"
#include "src/math/fadd.h"

static constexpr size_t DOUBLE_ROUNDS = 40;

float fadd_placeholder_binary(double x, double y) {
  return static_cast<float>(x + y);
}

int main() {
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float, double, LIBC_NAMESPACE::fadd,
                                  fadd_placeholder_binary, DOUBLE_ROUNDS,
                                  "fadd_perf.log")
  return 0;
}
