//===-- Differential test for sqrtf128
//----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SingleInputSingleOutputPerf.h"

#include "src/__support/FPUtil/sqrt.h"
#include "src/math/sqrtf128.h"

float128 sqrtf128_placeholder(float128 x) {
  return LIBC_NAMESPACE::fputil::sqrt<float128>(x);
}

SINGLE_INPUT_SINGLE_OUTPUT_PERF(float128, LIBC_NAMESPACE::sqrtf128,
                                ::sqrtf128_placeholder, "sqrtf128_perf.log")
