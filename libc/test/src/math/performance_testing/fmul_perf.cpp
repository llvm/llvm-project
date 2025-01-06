//===-- Performance test for the fmul function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BinaryOpSingleOutputPerf.h"
#include "src/__support/FPUtil/generic/mul.h"
#include "src/math/fmul.h"

static constexpr size_t DOUBLE_ROUNDS = 40;

float fmul_placeholder_binary(double x, double y) {
  return LIBC_NAMESPACE::fputil::generic::mul<float>(x, y);
}

int main() {
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float, double, LIBC_NAMESPACE::fmul,
                                  fmul_placeholder_binary, DOUBLE_ROUNDS,
                                  "fmul_perf.log")
  return 0;
}
