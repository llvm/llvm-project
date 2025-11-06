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

LLVM_LIBC_FUNCTION(int, rand, (void)) {
  unsigned long orig = rand_next.load(cpp::MemoryOrder::RELAXED);

  // An implementation of the xorshift64star pseudo random number generator.
  // This is a good general purpose generator for most non-cryptographics
  // applications.
  if constexpr (sizeof(void *) == sizeof(uint64_t)) {
    for (;;) {
      unsigned long x = orig;
      x ^= x >> 12;
      x ^= x << 25;
      x ^= x >> 27;
      if (rand_next.compare_exchange_strong(orig, x, cpp::MemoryOrder::ACQUIRE,
                                            cpp::MemoryOrder::RELAXED))
        return static_cast<int>((x * 0x2545F4914F6CDD1Dul) >> 32) & RAND_MAX;
      sleep_briefly();
    }
  } else {
    // This is the xorshift32 pseudo random number generator, slightly different
    // from the 64-bit star version above, as the previous version fails to
    // generate uniform enough LSB in 32-bit systems.
    for (;;) {
      unsigned long x = orig;
      x ^= x >> 13;
      x ^= x << 27;
      x ^= x >> 5;
      if (rand_next.compare_exchange_strong(orig, x, cpp::MemoryOrder::ACQUIRE,
                                            cpp::MemoryOrder::RELAXED))
        return static_cast<int>(x * 1597334677ul) & RAND_MAX;
      sleep_briefly();
    }
  }
  __builtin_unreachable();
}

} // namespace LIBC_NAMESPACE_DECL
