//===-- Performance test for fmodf16 --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BinaryOpSingleOutputPerf.h"

#include "src/__support/FPUtil/generic/FMod.h"
#include "src/__support/macros/properties/types.h"

#include <stdint.h>

#define FMOD_FUNC(U) (LIBC_NAMESPACE::fputil::generic::FMod<float16, U>::eval)

int main() {
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float16, float16, FMOD_FUNC(uint16_t),
                                  FMOD_FUNC(uint32_t), 5000,
                                  "fmodf16_u16_vs_u32_perf.log")

  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float16, float16, FMOD_FUNC(uint16_t),
                                  FMOD_FUNC(uint64_t), 5000,
                                  "fmodf16_u16_vs_u64_perf.log")
  return 0;
}
