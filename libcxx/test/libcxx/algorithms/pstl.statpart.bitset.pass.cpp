//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: std-at-least-c++17
// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// struct __bitset_view;

// struct __sharded_dynamic_bitset;

// optional<__sharded_dynamic_bitset> __make_sharded_dynamic_bitset(size_t, size_t) noexcept;

#include <__pstl/backends/statpart/bitset.h>
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <new>
#include <optional>
#include <utility>

using std::__pstl::__statpart::__bitset_view;
using std::__pstl::__statpart::__make_sharded_dynamic_bitset;
using std::__pstl::__statpart::__sharded_dynamic_bitset;

constexpr std::size_t bits_per_word      = CHAR_BIT * sizeof(std::size_t);
constexpr std::size_t words_per_line     = std::hardware_destructive_interference_size / sizeof(std::size_t);
constexpr std::size_t bits_per_cacheline = std::hardware_destructive_interference_size * CHAR_BIT;

int main(int, char**) {
  // Test __bitset_view
  {
    { // Set, test and reset individual bits
      std::size_t words[3] = {0, 0, 0};
      __bitset_view v{words, 3};
      const std::size_t positions[] = {
          0, 1, bits_per_word - 1, bits_per_word, bits_per_word + 1, 3 * bits_per_word - 1};
      for (std::size_t pos : positions) {
        assert(!v.__test(pos));
        v.__set(pos);
        assert(v.__test(pos));
      }
      assert(words[0] == (std::size_t(1) | std::size_t(2) | (std::size_t(1) << (bits_per_word - 1))));
      assert(words[1] == (std::size_t(1) | std::size_t(2)));
      assert(words[2] == (std::size_t(1) << (bits_per_word - 1)));
      for (std::size_t pos = 0; pos < 3 * bits_per_word; ++pos) {
        assert(v.__test(pos) == (bool)std::count(std::begin(positions), std::end(positions), pos));
      }
      for (std::size_t pos : positions) {
        v.__reset(pos);
        assert(!v.__test(pos));
      }
      assert(words[0] == 0 && words[1] == 0 && words[2] == 0);
    }
    { // Setting an already set bit and resetting an already reset bit are no-ops
      std::size_t words[1] = {0};
      __bitset_view v{words, 1};
      v.__set(5);
      v.__set(5);
      assert(words[0] == (std::size_t(1) << 5));
      v.__reset(6);
      assert(words[0] == (std::size_t(1) << 5));
      v.__reset(5);
      v.__reset(5);
      assert(words[0] == 0);
    }
    { // Resetting a single bit leaves the neighbours intact
      std::size_t words[1] = {~std::size_t(0)};
      __bitset_view v{words, 1};
      v.__reset(7);
      assert(words[0] == ~(std::size_t(1) << 7));
    }
    { // __reset() clears all words of the view and nothing beyond it
      std::size_t words[4] = {~std::size_t(0), ~std::size_t(0), ~std::size_t(0), ~std::size_t(0)};
      __bitset_view v{words + 1, 2};
      v.__reset();
      assert(words[0] == ~std::size_t(0));
      assert(words[1] == 0);
      assert(words[2] == 0);
      assert(words[3] == ~std::size_t(0));
    }
    { // __reset() on an empty view does nothing
      std::size_t word = ~std::size_t(0);
      __bitset_view v{&word, 0};
      v.__reset();
      assert(word == ~std::size_t(0));
    }
  }

  // Test the shard size rounding
  {
    { // A single bit occupies a full cache line
      std::optional<__sharded_dynamic_bitset> b = __make_sharded_dynamic_bitset(1, 1);
      assert(b.has_value());
      assert(b->__shards_count_ == 1);
      assert(b->__words_per_shard_ == words_per_line);
      assert(b->__shard(0).__words_ == words_per_line);
    }
    { // Exactly one cache line worth of bits
      std::optional<__sharded_dynamic_bitset> b = __make_sharded_dynamic_bitset(2, bits_per_cacheline);
      assert(b.has_value());
      assert(b->__words_per_shard_ == words_per_line);
    }
    { // One bit over a cache line spills into the next cache line
      std::optional<__sharded_dynamic_bitset> b = __make_sharded_dynamic_bitset(3, bits_per_cacheline + 1);
      assert(b.has_value());
      assert(b->__words_per_shard_ == 2 * words_per_line);
    }
  }

  // Test the shard layout
  {
    constexpr std::size_t shards              = 5;
    constexpr std::size_t bits                = 1000;
    std::optional<__sharded_dynamic_bitset> b = __make_sharded_dynamic_bitset(shards, bits);
    assert(b.has_value());
    const std::size_t words = b->__words_per_shard_;
    assert(words * bits_per_word >= bits);

    // Shards are adjacent, cover the whole allocation and don't overlap
    for (std::size_t i = 0; i < shards; ++i) {
      __bitset_view v = b->__shard(i);
      assert(v.__bits_ == b->__bits_.get() + i * words);
      assert(v.__words_ == words);
    }

#if _LIBCPP_HAS_ALIGNED_ALLOCATION
    // Each shard starts at a cache line boundary
    for (std::size_t i = 0; i < shards; ++i) {
      auto addr = reinterpret_cast<std::uintptr_t>(b->__shard(i).__bits_);
      assert(addr % std::hardware_destructive_interference_size == 0);
    }
#endif
  }

  return 0;
}
