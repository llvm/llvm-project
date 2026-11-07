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

// __hive_bitmap::__find_free(block): the lowest free slot. The block must have one.

#include <cassert>
#include <cstddef>
#include <hive>

#include "bitmap_fixture.h"
#include "test_macros.h"

// A block fresh from __reset gives slot 0.
template <std::size_t Cap>
constexpr void test_fresh_block() {
  Fixture<Cap> f;
  Bitmap::__reset(f.block);
  assert(Bitmap::__invariants(f.block));
  assert(Bitmap::__find_free(f.block) == 0);
}

// The lowest free slot, below higher live ones.
constexpr void test_lowest_free_slot() {
  Fixture<200> f;
  f.set(0);
  f.set(1);
  f.set(2);
  f.set(5);
  f.finish();
  assert(Bitmap::__find_free(f.block) == 3);
}

// Word 0 full: the free summary moves the search to word 1.
constexpr void test_skips_full_word() {
  Fixture<200> f;
  for (std::size_t i = 0; i != 66; ++i) {
    f.set(i);
  }
  f.finish();
  assert(Bitmap::__find_free(f.block) == 66);
}

// Slot 199 is the only free valid bit of the last word; the padding bits above it are zero too and must not be found.
constexpr void test_last_word_padding() {
  Fixture<200> f;
  for (std::size_t i = 0; i != 199; ++i) {
    f.set(i);
  }
  f.finish();
  assert(Bitmap::__find_free(f.block) == 199);
}

// Capacity 65: word 1 has one valid slot.
constexpr void test_single_slot_word() {
  Fixture<65> f;
  for (std::size_t i = 0; i != 64; ++i) {
    f.set(i);
  }
  f.finish();
  assert(Bitmap::__find_free(f.block) == 64);
}

// Word 63 is the last live word and bit 63 of it the last slot.
constexpr void test_largest_block() {
  Fixture<4096> f;
  for (std::size_t i = 0; i != 4095; ++i) {
    f.set(i);
  }
  f.finish();
  assert(Bitmap::__find_free(f.block) == 4095);
}

constexpr bool test() {
  test_fresh_block<1>();
  test_fresh_block<64>();
  test_fresh_block<200>();
  test_fresh_block<4096>();
  test_lowest_free_slot();
  test_skips_full_word();
  test_last_word_padding();
  test_single_slot_word();
  test_largest_block();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
