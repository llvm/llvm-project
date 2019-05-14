//===-- random.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GWP_ASAN_RANDOM_H_
#define GWP_ASAN_RANDOM_H_

#include <cstdint>

namespace gwp_asan {
// xorshift (32-bit output), extremely fast PRNG that uses arithmetic operations
// only. Seeded using walltime.
uint32_t getRandomUnsigned32();
} // namespace gwp_asan

#endif // GWP_ASAN_RANDOM_H_
