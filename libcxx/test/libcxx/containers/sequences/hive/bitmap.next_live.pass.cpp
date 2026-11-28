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

// __hive_bitmap::__next_live(block, i): the first live slot at or after i, or the capacity when there is none.
// i may equal the capacity, never exceed it.

#include <cassert>
#include <cstddef>
#include <hive>

#include "bitmap_fixture.h"
#include "test_macros.h"

// Empty block: every query lands on the capacity.
constexpr void test_empty_block() {
  Fixture<200> f;
  f.finish();
  assert(Bitmap::__next_live(f.block, 0) == 200);
  assert(Bitmap::__next_live(f.block, 199) == 200);
  assert(Bitmap::__next_live(f.block, 200) == 200);
}

// Bits in words 0, 1 and 3; word 2 is skipped through the summary.
constexpr void test_skips_empty_words() {
  Fixture<200> f;
  f.set(5);
  f.set(70);
  f.set(199);
  f.finish();
  assert(Bitmap::__next_live(f.block, 0) == 5);
  assert(Bitmap::__next_live(f.block, 5) == 5);
  assert(Bitmap::__next_live(f.block, 6) == 70);
  assert(Bitmap::__next_live(f.block, 63) == 70);
  assert(Bitmap::__next_live(f.block, 64) == 70);
  assert(Bitmap::__next_live(f.block, 70) == 70);
  assert(Bitmap::__next_live(f.block, 71) == 199);
  assert(Bitmap::__next_live(f.block, 128) == 199);
  assert(Bitmap::__next_live(f.block, 199) == 199);
  assert(Bitmap::__next_live(f.block, 200) == 200);
}

// Capacity a multiple of 64: the last live word is full and the sentinel word stays empty.
constexpr void test_capacity_multiple_of_word() {
  Fixture<128> f;
  f.set(0);
  f.set(127);
  f.finish();
  assert(Bitmap::__next_live(f.block, 0) == 0);
  assert(Bitmap::__next_live(f.block, 1) == 127);
  assert(Bitmap::__next_live(f.block, 127) == 127);
  assert(Bitmap::__next_live(f.block, 128) == 128);
}

// Word 63 is the last live word: nothing may be looked up past it.
constexpr void test_largest_block() {
  Fixture<4096> f;
  f.set(4032);
  f.set(4095);
  f.finish();
  assert(Bitmap::__next_live(f.block, 0) == 4032);
  assert(Bitmap::__next_live(f.block, 4033) == 4095);
  assert(Bitmap::__next_live(f.block, 4095) == 4095);
  assert(Bitmap::__next_live(f.block, 4096) == 4096);
}

// Every slot live: the query at the capacity lands on the padding bits of the last word (200) or on the zero word
// (1, 128, 4096), which read 0.
template <std::size_t Cap>
constexpr void test_full_block() {
  Fixture<Cap> f;
  f.set_all();
  f.finish();
  assert(Bitmap::__next_live(f.block, 0) == 0);
  assert(Bitmap::__next_live(f.block, Cap - 1) == Cap - 1);
  assert(Bitmap::__next_live(f.block, Cap) == Cap);
}

constexpr bool test() {
  test_empty_block();
  test_skips_empty_words();
  test_capacity_multiple_of_word();
  test_largest_block();
  test_full_block<1>();
  test_full_block<128>();
  test_full_block<200>();
  test_full_block<4096>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
