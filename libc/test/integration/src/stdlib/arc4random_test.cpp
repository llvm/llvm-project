//===-- Test for arc4random functions -----------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/arc4random.h"
#include "src/stdlib/arc4random_buf.h"
#include "src/stdlib/arc4random_uniform.h"
#include "test/IntegrationTest/test.h"

using namespace LIBC_NAMESPACE;

void arc4random_smoke_test() {
  constexpr static size_t NUM_ITERATIONS = 100;

  for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
    uint32_t value1 = arc4random();
    uint32_t value2 = arc4random();

    // It is unlikely that both values are 0.
    if (value1 == 0 && value2 == 0)
      __builtin_trap();
  }
}

void arc4random_buf_smoke_test() {
  constexpr static size_t NUM_ITERATIONS = 50;

  for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
    char buffer1[256] = {0};
    char buffer2[256] = {0};

    arc4random_buf(buffer1, sizeof(buffer1));
    arc4random_buf(buffer2, sizeof(buffer2));

    // Basic sanity check - buffers should be different (very unlikely to be
    // same) Count non-zero bytes to ensure we got some random data
    size_t non_zero_count1 = 0;
    size_t non_zero_count2 = 0;

    for (size_t j = 0; j < sizeof(buffer1); ++j) {
      if (buffer1[j] != 0)
        ++non_zero_count1;
      if (buffer2[j] != 0)
        ++non_zero_count2;
    }

    // It is extremely unlikely to get all zeros from arc4random_buf
    if (non_zero_count1 == 0 && non_zero_count2 == 0)
      __builtin_trap();
  }
}

void arc4random_uniform_smoke_test() {
  constexpr static size_t NUM_ITERATIONS = 100;

  for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
    uint32_t upper_bound = 1000000;
    uint32_t value1 = arc4random_uniform(upper_bound);
    uint32_t value2 = arc4random_uniform(upper_bound);

    // Basic sanity check - values should be less than upper_bound
    if (value1 >= upper_bound || value2 >= upper_bound)
      __builtin_trap(); // Values should be less than upper_bound

    // It is extremely unlikely to get both 0 from arc4random_uniform.
    if (value1 == 0 && value2 == 0)
      __builtin_trap();
  }
}

TEST_MAIN() {
  arc4random_smoke_test();
  arc4random_buf_smoke_test();
  arc4random_uniform_smoke_test();
  return 0;
}
