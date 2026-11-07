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

// __hive_iterator ++ and -- over hand-built blocks: within a word, across words through the summary, and across
// blocks; from and to end(); the pre and post forms.

#include <cassert>
#include <cstddef>
#include <hive>

#include "bitmap_fixture.h"
#include "test_macros.h"

using It  = std::__hive_iterator<int, void*, false>;
using CIt = std::__hive_iterator<int, void*, true>;

// The iterator on slot `index` of `block`.
template <class Iter = It>
Iter at(Block& block, std::size_t index) {
  return std::__make_hive_iterator<Iter>(&block, index);
}

// Block A (capacity 200) has slots 5, 70 and 199; block B (capacity 128) has slots 0 and 127; A precedes B.
struct Chain {
  Fixture<200> a;
  Fixture<128> b;

  Chain() {
    a.set(5);
    a.set(70);
    a.set(199);
    a.finish();
    b.set(0);
    b.set(127);
    b.finish();
    a.block.__next_ = &b.block;
    b.block.__prev_ = &a.block;
    for (std::size_t i = 0; i != 200; ++i) {
      a.slots[i] = 1000 + static_cast<int>(i);
    }
    for (std::size_t i = 0; i != 128; ++i) {
      b.slots[i] = 2000 + static_cast<int>(i);
    }
  }
};

// Forward: the rest of the word, the next word through the summary, the next block, then end().
void test_increment() {
  Chain c;
  It it = at(c.a.block, 5);
  assert(*it == 1005);
  ++it;
  assert(it == at(c.a.block, 70));
  assert(*it == 1070);
  ++it;
  assert(it == at(c.a.block, 199));
  ++it;
  assert(it == at(c.b.block, 0));
  assert(*it == 2000);
  ++it;
  assert(it == at(c.b.block, 127));
  assert(*it == 2127);
  ++it;
  assert(it == at(c.b.block, 128));
}

// Backward from end(): the last slot of the last block, the previous word, the previous block, then the first slot.
void test_decrement() {
  Chain c;
  It it = at(c.b.block, 128);
  --it;
  assert(it == at(c.b.block, 127));
  --it;
  assert(it == at(c.b.block, 0));
  --it;
  assert(it == at(c.a.block, 199));
  --it;
  assert(it == at(c.a.block, 70));
  --it;
  assert(it == at(c.a.block, 5));
  assert(*it == 1005);
}

// The post forms return the old position.
void test_post_forms() {
  Chain c;
  It it  = at(c.a.block, 70);
  It old = it++;
  assert(old == at(c.a.block, 70));
  assert(it == at(c.a.block, 199));
  old = it--;
  assert(old == at(c.a.block, 199));
  assert(it == at(c.a.block, 70));
}

// A dense block: every step is the fast path, and the walk ends at the capacity.
void test_dense_block() {
  Fixture<200> f;
  f.set_all();
  f.finish();
  for (std::size_t i = 0; i != 200; ++i) {
    f.slots[i] = static_cast<int>(i);
  }
  It it = at(f.block, 0);
  for (int i = 0; i != 200; ++i) {
    assert(*it == i);
    ++it;
  }
  assert(it == at(f.block, 200));
  for (int i = 199; i != -1; --i) {
    --it;
    assert(*it == i);
  }
  assert(it == at(f.block, 0));
}

// -> and the const iterator read the same slot; the non-const iterator converts and compares both ways.
void test_arrow_and_const() {
  Chain c;
  It it = at(c.a.block, 70);
  assert(it.operator->() == &c.a.slots[70]);
  CIt cit = it;
  assert(cit == it);
  assert(it == cit);
  assert(*cit == 1070);
  assert(cit.operator->() == &c.a.slots[70]);
  ++cit;
  assert(cit == at<CIt>(c.a.block, 199));
}

int main(int, char**) {
  test_increment();
  test_decrement();
  test_post_forms();
  test_dense_block();
  test_arrow_and_const();

  return 0;
}
