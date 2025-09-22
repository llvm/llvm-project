//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// size_type size() const noexcept;

#include <cassert>
#include <deque>
#include <flat_set>
#include <functional>
#include <vector>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer>
void test_one() {
  using M = std::flat_multiset<int, std::less<int>, KeyContainer>;
  using S = typename M::size_type;
  {
    const M m = {1, 1, 4, 5, 5};
    ASSERT_SAME_TYPE(decltype(m.size()), S);
    ASSERT_NOEXCEPT(m.size());
    assert(m.size() == 5);
  }
  {
    const M m = {1};
    ASSERT_SAME_TYPE(decltype(m.size()), S);
    ASSERT_NOEXCEPT(m.size());
    assert(m.size() == 1);
  }
  {
    const M m;
    ASSERT_SAME_TYPE(decltype(m.size()), S);
    ASSERT_NOEXCEPT(m.size());
    assert(m.size() == 0);
  }
  {
    M m;
    S s = 500000;
    for (std::size_t i = 0u; i < s; ++i) {
      m.emplace(i);
      m.emplace(i);
    }
    ASSERT_SAME_TYPE(decltype(m.size()), S);
    ASSERT_NOEXCEPT(m.size());
    assert(m.size() == 2 * s);
  }
}

void test() {
  test_one<std::vector<int>>();
  test_one<std::deque<int>>();
  test_one<MinSequenceContainer<int>>();
  test_one<std::vector<int, min_allocator<int>>>();
}

int main(int, char**) {
  test();

  return 0;
}
