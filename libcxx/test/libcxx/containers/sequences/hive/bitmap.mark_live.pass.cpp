//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive

// <hive>

// __hive_bitmap::__mark_live(block, i): sets bit i and the summaries. The caller counts __size_.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <hive>

#include "bitmap_fixture.h"
#include "test_macros.h"

// Marks as emplace will: the bit, then the size, with the invariants checked after each step.
template <std::size_t Cap>
constexpr void mark(Fixture<Cap>& f, std::size_t i) {
  Bitmap::__mark_live(f.block, i);
  ++f.block.__size_;
  assert(Bitmap::__invariants(f.block));
}

// First slot of a fresh block: live summary bit 0, free summary untouched.
constexpr void test_first_slot() {
  Fixture<200> f;
  Bitmap::__reset(f.block);
  mark(f, 0);
  assert(f.words[0] == 1);
  assert(f.block.__live_summary_ == 0b1);
  assert(f.block.__free_summary_ == 0b1111);
}

// A word loses its free-summary bit when it fills, not before.
constexpr void test_word_fills() {
  Fixture<200> f;
  Bitmap::__reset(f.block);
  for (std::size_t i = 0; i != 63; ++i) {
    mark(f, i);
  }
  assert(f.block.__free_summary_ == 0b1111);
  mark(f, 63);
  assert(f.words[0] == ~std::uint64_t(0));
  assert(f.block.__live_summary_ == 0b1);
  assert(f.block.__free_summary_ == 0b1110);
}

// The partial last word fills at its 8 valid bits, not at 64.
constexpr void test_partial_word_fills() {
  Fixture<200> f;
  Bitmap::__reset(f.block);
  for (std::size_t i = 192; i != 199; ++i) {
    mark(f, i);
  }
  assert(f.block.__free_summary_ == 0b1111);
  mark(f, 199);
  assert(f.words[3] == 0xff);
  assert(f.block.__live_summary_ == 0b1000);
  assert(f.block.__free_summary_ == 0b0111);
}

// One slot: it fills the block.
constexpr void test_capacity_one() {
  Fixture<1> f;
  Bitmap::__reset(f.block);
  mark(f, 0);
  assert(f.words[0] == 1);
  assert(f.block.__live_summary_ == 1);
  assert(f.block.__free_summary_ == 0);
}

// Capacity 64: one live word that fills at all 64 bits; the zero word stays 0.
constexpr void test_capacity_multiple_of_word() {
  Fixture<64> f;
  Bitmap::__reset(f.block);
  for (std::size_t i = 0; i != 63; ++i) {
    mark(f, i);
  }
  assert(f.block.__free_summary_ == 1);
  mark(f, 63);
  assert(f.block.__free_summary_ == 0);
  assert(f.words[1] == 0);
}

// Bit 63 of word 63 (the shift by 63) and the last free-summary bit.
constexpr void test_largest_block() {
  Fixture<4096> f;
  Bitmap::__reset(f.block);
  mark(f, 4095);
  assert(f.words[63] == std::uint64_t(1) << 63);
  assert(f.block.__live_summary_ == std::uint64_t(1) << 63);
  assert(f.block.__free_summary_ == ~std::uint64_t(0));
  for (std::size_t i = 4032; i != 4095; ++i) {
    mark(f, i);
  }
  assert(f.block.__free_summary_ == ~std::uint64_t(0) >> 1);
  assert(f.words[64] == 0);
}

// Emplace's loop, find then mark, hands out the slots in order until the block is full.
constexpr void test_find_then_mark() {
  Fixture<200> f;
  Bitmap::__reset(f.block);
  for (std::size_t i = 0; i != 200; ++i) {
    assert(Bitmap::__find_free(f.block) == i);
    mark(f, i);
  }
  assert(f.block.__free_summary_ == 0);
  assert(f.block.__live_summary_ == 0b1111);
  assert(f.block.__size_ == 200);
}

constexpr bool test() {
  test_first_slot();
  test_word_fills();
  test_partial_word_fills();
  test_capacity_one();
  test_capacity_multiple_of_word();
  test_largest_block();
  test_find_then_mark();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
