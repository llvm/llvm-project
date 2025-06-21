//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// size_type size() const noexcept;

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <type_traits>
#include <vector>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using M = std::flat_map<int, char, std::less<int>, KeyContainer, ValueContainer>;
  {
    const M m = {{1, 'a'}, {1, 'b'}, {4, 'd'}, {5, 'e'}, {5, 'h'}};
    ASSERT_SAME_TYPE(decltype(m.size()), std::size_t);
    ASSERT_NOEXCEPT(m.size());
    assert(m.size() == 3);
  }
  {
    const M m = {{1, 'a'}};
    ASSERT_SAME_TYPE(decltype(m.size()), std::size_t);
    ASSERT_NOEXCEPT(m.size());
    assert(m.size() == 1);
  }
  {
    const M m;
    ASSERT_SAME_TYPE(decltype(m.size()), std::size_t);
    ASSERT_NOEXCEPT(m.size());
    assert(m.size() == 0);
  }
  {
    M m;
    std::size_t s = TEST_IS_CONSTANT_EVALUATED ? 100 : 1000000;
    for (auto i = 0u; i < s; ++i) {
      m.emplace(i, 'a');
    }
    ASSERT_SAME_TYPE(decltype(m.size()), std::size_t);
    ASSERT_NOEXCEPT(m.size());
    assert(m.size() == s);
  }
}

constexpr bool test() {
  test<std::vector<int>, std::vector<char>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test<std::deque<int>, std::vector<char>>();
  }
  test<MinSequenceContainer<int>, MinSequenceContainer<char>>();
  test<std::vector<int, min_allocator<int>>, std::vector<char, min_allocator<char>>>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
