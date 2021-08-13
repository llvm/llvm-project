//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// take_view() requires default_initializable<V> = default;
// constexpr take_view(V base, range_difference_t<V> count);

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "test_range.h"
#include "types.h"

int globalBuffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

template<bool IsDefaultCtorable>
struct DefaultConstructible : std::ranges::view_base {
  DefaultConstructible() requires IsDefaultCtorable = default;
  DefaultConstructible(int*);
  int* begin();
  sentinel_wrapper<int*> end();
};

struct SizedRandomAccessViewToGlobal : std::ranges::view_base {
  RandomAccessIter begin() { return RandomAccessIter(globalBuffer); }
  RandomAccessIter begin() const { return RandomAccessIter(globalBuffer); }
  sentinel_wrapper<RandomAccessIter> end() {
    return sentinel_wrapper<RandomAccessIter>{RandomAccessIter(globalBuffer + 8)};
  }
  sentinel_wrapper<RandomAccessIter> end() const {
    return sentinel_wrapper<RandomAccessIter>{RandomAccessIter(globalBuffer + 8)};
  }
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    std::ranges::take_view<CopyableView> tv(CopyableView{buffer}, 0);
    assert(tv.base().ptr_ == buffer);
    assert(tv.begin() == tv.end()); // Checking we have correct size.
  }

  {
    std::ranges::take_view<ContiguousView> tv(ContiguousView{buffer}, 1);
    assert(std::move(tv).base().ptr_ == buffer);
    assert(std::ranges::next(tv.begin(), 1) == tv.end()); // Checking we have correct size.
  }

  {
    const std::ranges::take_view<CopyableView> tv(CopyableView{buffer}, 2);
    assert(tv.base().ptr_ == buffer);
    assert(std::ranges::next(tv.begin(), 2) == tv.end()); // Checking we have correct size.
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  // Tests for the default ctor.
  static_assert( std::default_initializable<DefaultConstructible<true>>);
  static_assert(!std::default_initializable<DefaultConstructible<false>>);

  std::ranges::take_view<SizedRandomAccessViewToGlobal> tv;
  assert(*tv.base().begin() == 1);
  assert(tv.size() == 0);

  return 0;
}
