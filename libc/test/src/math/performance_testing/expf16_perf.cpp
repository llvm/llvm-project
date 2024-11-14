//===-- Performance test for expf16 ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SingleInputSingleOutputPerf.h"

#include "src/math/expf16.h"

// LLVM libc might be the only libc implementation with support for float16 math
// functions currently. We can't compare our float16 functions against the
// system libc, so we compare them against this placeholder function.
static float16 placeholderf16(float16 x) { return x; }

int main() {
  SINGLE_INPUT_SINGLE_OUTPUT_PERF_EX(float16, LIBC_NAMESPACE::expf16,
                                     ::placeholderf16, 20'000,
                                     "expf16_perf.log")
}
