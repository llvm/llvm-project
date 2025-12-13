//===-- Differential test for hypotf16 ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PerfTest.h"

#include "src/__support/FPUtil/Hypot.h"
#include "src/math/hypotf16.h"

int main() {
  BINARY_INPUT_SINGLE_OUTPUT_PERF(float16, float16, LIBC_NAMESPACE::hypotf16,
                                  LIBC_NAMESPACE::fputil::hypot<float16>,
                                  "hypotf16_perf.log")
  return 0;
}
