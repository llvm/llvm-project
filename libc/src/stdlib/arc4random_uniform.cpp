//===-- Implementation of arc4random_uniform --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/arc4random_uniform.h"
#include "src/__support/CPP/optional.h"
#include "src/stdlib/linux/vdso_rng.h"

namespace LIBC_NAMESPACE_DECL {
// Bounded random number generation by Daniel Lemire.
// See also
// https://lemire.me/blog/2024/08/17/faster-random-integer-generation-with-batching/
// https://arxiv.org/abs/1805.10941
// Fast Random Integer Generation in an Interval
uint32_t arc4random_uniform(uint32_t upper_bound) {
  auto gen = [] {
    using namespace vdso_rng;
    uint32_t result = 0;
    if (cpp::optional<LocalState::Guard> guard = local_state.get())
      guard->fill(&result, sizeof(result));
    else
      fallback_rng_fill(&result, sizeof(result));
    return result;
  };
  uint64_t rand32 = gen();
  uint64_t multiplied = rand32 * upper_bound;
  uint32_t leftover = static_cast<uint32_t>(multiplied);
  if (leftover < upper_bound) {
    uint32_t threshold = -upper_bound % upper_bound;
    while (leftover < threshold) {
      rand32 = gen();
      multiplied = rand32 * upper_bound;
      leftover = static_cast<uint32_t>(multiplied);
    }
  }
  return static_cast<uint32_t>(multiplied >> 32);
}
} // namespace LIBC_NAMESPACE_DECL
