//===-- Unittests for rand ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/rand.h"
#include "src/stdlib/srand.h"
#include "test/UnitTest/Test.h"

#include <stddef.h>
#include <stdlib.h>

TEST(LlvmLibcRandTest, UnsetSeed) {
  for (size_t i = 0; i < 1000; ++i) {
    int val = __llvm_libc::rand();
    ASSERT_GE(val, 0);
    ASSERT_LE(val, RAND_MAX);
  }
}

TEST(LlvmLibcRandTest, SetSeed) {
  const unsigned int SEED = 12344321;
  __llvm_libc::srand(SEED);
  const size_t NUM_RESULTS = 10;
  int results[NUM_RESULTS];
  for (size_t i = 0; i < NUM_RESULTS; ++i) {
    results[i] = __llvm_libc::rand();
    ASSERT_GE(results[i], 0);
    ASSERT_LE(results[i], RAND_MAX);
  }

  // If the seed is set to the same value, it should give the same sequence.
  __llvm_libc::srand(SEED);

  for (size_t i = 0; i < NUM_RESULTS; ++i) {
    int val = __llvm_libc::rand();
    EXPECT_EQ(results[i], val);
  }
}
