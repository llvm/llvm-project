//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_LIBCXX_CONTAINERS_SEQUENCES_HIVE_BITMAP_FIXTURE_H
#define TEST_LIBCXX_CONTAINERS_SEQUENCES_HIVE_BITMAP_FIXTURE_H

// A block built by hand, so the bit helpers are tested without emplace.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <hive>

using Block  = std::__hive_block<int, void*>;
using Bitmap = std::__hive_bitmap<Block>;

template <std::size_t Cap>
struct Fixture {
  std::uint64_t words[Cap / 64 + 1] = {};
  int slots[Cap]                    = {};
  Block block{words, slots, Cap};

  // Sets bit i, the live summary and the size by hand; finish() derives the free summary.
  constexpr void set(std::size_t i) {
    words[i / 64] |= std::uint64_t(1) << (i % 64);
    block.__live_summary_ |= std::uint64_t(1) << (i / 64);
    ++block.__size_;
  }

  constexpr void set_all() {
    for (std::size_t i = 0; i != Cap; ++i) {
      set(i);
    }
  }

  constexpr void finish() {
    for (std::size_t w = 0; w != (Cap + 63) / 64; ++w) {
      const std::size_t valid_bits = Cap - w * 64 < 64 ? Cap - w * 64 : 64;
      const std::uint64_t valid    = valid_bits == 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << valid_bits) - 1;
      block.__free_summary_ |= std::uint64_t((~words[w] & valid) != 0) << w;
    }
    assert(Bitmap::__invariants(block));
  }
};

#endif // TEST_LIBCXX_CONTAINERS_SEQUENCES_HIVE_BITMAP_FIXTURE_H
