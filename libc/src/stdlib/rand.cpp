//===-- Implementation of rand --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/rand.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/sleep.h"
#include "src/stdlib/rand_util.h"

namespace LIBC_NAMESPACE_DECL {

// Default random key as used in https://squaresrng.wixsite.com/rand
static constexpr uint64_t RANDOM_KEY = 0x548c9decbce65297;

LLVM_LIBC_FUNCTION(int, rand, (void)) {
  // Based on Squares: A Fast Counter-Based RNG
  // https://arxiv.org/pdf/2004.06278
  uint64_t counter = static_cast<uint64_t>(rand_next.fetch_add(1));
  uint64_t x = counter * RANDOM_KEY, y = counter * RANDOM_KEY;
  uint64_t z = y + RANDOM_KEY;
  x = x * x + y;
  x = (x >> 32) | (x << 32);
  x = x * x + z;
  x = (x >> 32) | (x << 32); 
  x = x * x + y;
  x = (x >> 32) | (x << 32); 
  uint64_t result = (x * x + z) >> 32;
  // project into range
  return static_cast<int>((result % RAND_MAX + RAND_MAX) % RAND_MAX);
}

} // namespace LIBC_NAMESPACE_DECL
