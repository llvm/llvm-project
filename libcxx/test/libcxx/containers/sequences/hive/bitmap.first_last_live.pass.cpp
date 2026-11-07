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

// __hive_bitmap::__first_live(block) and __last_live(block): the first and last live slots, or the capacity when the
// block is empty.

#include <cassert>
#include <cstddef>
#include <hive>

#include "bitmap_fixture.h"
#include "test_macros.h"

// Empty block: both land on the capacity.
constexpr void test_empty_block() {
  Fixture<200> f;
  f.finish();
  assert(Bitmap::__first_live(f.block) == 200);
  assert(Bitmap::__last_live(f.block) == 200);
}

// One live slot in the middle: it is both.
constexpr void test_one_slot() {
  Fixture<200> f;
  f.set(70);
  f.finish();
  assert(Bitmap::__first_live(f.block) == 70);
  assert(Bitmap::__last_live(f.block) == 70);
}

// The ends of the block.
constexpr void test_ends() {
  Fixture<200> f;
  f.set(0);
  f.set(199);
  f.finish();
  assert(Bitmap::__first_live(f.block) == 0);
  assert(Bitmap::__last_live(f.block) == 199);
}

// Every slot live, including the capacities where the last query reads the zero word.
template <std::size_t Cap>
constexpr void test_full_block() {
  Fixture<Cap> f;
  f.set_all();
  f.finish();
  assert(Bitmap::__first_live(f.block) == 0);
  assert(Bitmap::__last_live(f.block) == Cap - 1);
}

constexpr bool test() {
  test_empty_block();
  test_one_slot();
  test_ends();
  test_full_block<1>();
  test_full_block<64>();
  test_full_block<4096>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
