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

// __hive_bitmap::__prev_live(block, i): the last live slot at or before i, or the capacity when there is none.
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
  assert(Bitmap::__prev_live(f.block, 0) == 200);
  assert(Bitmap::__prev_live(f.block, 199) == 200);
  assert(Bitmap::__prev_live(f.block, 200) == 200);
}

// Bits in words 0, 1 and 3; word 2 is skipped through the summary.
constexpr void test_skips_empty_words() {
  Fixture<200> f;
  f.set(5);
  f.set(70);
  f.set(199);
  f.finish();
  assert(Bitmap::__prev_live(f.block, 200) == 199);
  assert(Bitmap::__prev_live(f.block, 199) == 199);
  assert(Bitmap::__prev_live(f.block, 198) == 70);
  assert(Bitmap::__prev_live(f.block, 128) == 70);
  assert(Bitmap::__prev_live(f.block, 127) == 70);
  assert(Bitmap::__prev_live(f.block, 70) == 70);
  assert(Bitmap::__prev_live(f.block, 69) == 5);
  assert(Bitmap::__prev_live(f.block, 64) == 5);
  assert(Bitmap::__prev_live(f.block, 63) == 5);
  assert(Bitmap::__prev_live(f.block, 5) == 5);
  assert(Bitmap::__prev_live(f.block, 4) == 200);
  assert(Bitmap::__prev_live(f.block, 0) == 200);
}

// Capacity a multiple of 64: the query at the capacity reads the zero word and falls back to the summary.
constexpr void test_capacity_multiple_of_word() {
  Fixture<128> f;
  f.set(0);
  f.set(127);
  f.finish();
  assert(Bitmap::__prev_live(f.block, 128) == 127);
  assert(Bitmap::__prev_live(f.block, 127) == 127);
  assert(Bitmap::__prev_live(f.block, 126) == 0);
  assert(Bitmap::__prev_live(f.block, 0) == 0);
}

// Word 63 is the last live word; the query at 4096 reads word 64, the zero word.
constexpr void test_largest_block() {
  Fixture<4096> f;
  f.set(0);
  f.set(4032);
  f.set(4095);
  f.finish();
  assert(Bitmap::__prev_live(f.block, 4096) == 4095);
  assert(Bitmap::__prev_live(f.block, 4095) == 4095);
  assert(Bitmap::__prev_live(f.block, 4094) == 4032);
  assert(Bitmap::__prev_live(f.block, 4031) == 0);
  assert(Bitmap::__prev_live(f.block, 1) == 0);
}

// One slot: empty, then live.
constexpr void test_capacity_one() {
  Fixture<1> e;
  e.finish();
  assert(Bitmap::__prev_live(e.block, 0) == 1);
  assert(Bitmap::__prev_live(e.block, 1) == 1);

  Fixture<1> f;
  f.set(0);
  f.finish();
  assert(Bitmap::__prev_live(f.block, 0) == 0);
  assert(Bitmap::__prev_live(f.block, 1) == 0);
}

// Every slot live: the query at the capacity lands on the padding bits of the last word (200) or on the zero word
// (64, 128, 4096), which read 0, and the summary names the last word.
template <std::size_t Cap>
constexpr void test_full_block() {
  Fixture<Cap> f;
  f.set_all();
  f.finish();
  assert(Bitmap::__prev_live(f.block, Cap) == Cap - 1);
  assert(Bitmap::__prev_live(f.block, Cap - 1) == Cap - 1);
  assert(Bitmap::__prev_live(f.block, 0) == 0);
}

constexpr bool test() {
  test_empty_block();
  test_skips_empty_words();
  test_capacity_multiple_of_word();
  test_largest_block();
  test_capacity_one();
  test_full_block<64>();
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
