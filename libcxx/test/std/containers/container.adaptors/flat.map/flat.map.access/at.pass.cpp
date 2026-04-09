//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

//       mapped_type& at(const key_type& k);
// const mapped_type& at(const key_type& k) const;

#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "MinSequenceContainer.h"
#include "min_allocator.h"
#include "test_macros.h"

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using P = std::pair<int, double>;
  P ar[]  = {
      P(1, 1.5),
      P(2, 2.5),
      P(3, 3.5),
      P(4, 4.5),
      P(5, 5.5),
      P(7, 7.5),
      P(8, 8.5),
  };
  const int one = 1;
  {
    std::flat_map<int, double, std::less<int>, KeyContainer, ValueContainer> m(ar, ar + sizeof(ar) / sizeof(ar[0]));
    ASSERT_SAME_TYPE(decltype(m.at(one)), double&);
    assert(m.size() == 7);
    assert(m.at(one) == 1.5);
    m.at(1) = -1.5;
    assert(m.at(1) == -1.5);
    assert(m.at(2) == 2.5);
    assert(m.at(3) == 3.5);
    assert(m.at(4) == 4.5);
    assert(m.at(5) == 5.5);
#ifndef TEST_HAS_NO_EXCEPTIONS
    if (!TEST_IS_CONSTANT_EVALUATED) {
      try {
        TEST_IGNORE_NODISCARD m.at(6);
        assert(false);
      } catch (std::out_of_range&) {
      }
    }
#endif
    assert(m.at(7) == 7.5);
    assert(m.at(8) == 8.5);
    assert(m.size() == 7);
  }
  {
    const std::flat_map<int, double, std::less<int>, KeyContainer, ValueContainer> m(
        ar, ar + sizeof(ar) / sizeof(ar[0]));
    ASSERT_SAME_TYPE(decltype(m.at(one)), const double&);
    assert(m.size() == 7);
    assert(m.at(one) == 1.5);
    assert(m.at(2) == 2.5);
    assert(m.at(3) == 3.5);
    assert(m.at(4) == 4.5);
    assert(m.at(5) == 5.5);
#ifndef TEST_HAS_NO_EXCEPTIONS
    if (!TEST_IS_CONSTANT_EVALUATED) {
      try {
        TEST_IGNORE_NODISCARD m.at(6);
        assert(false);
      } catch (std::out_of_range&) {
      }
    }
#endif
    assert(m.at(7) == 7.5);
    assert(m.at(8) == 8.5);
    assert(m.size() == 7);
  }
}

constexpr bool test() {
  test<std::vector<int>, std::vector<double>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test<std::deque<int>, std::vector<double>>();
  }
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
