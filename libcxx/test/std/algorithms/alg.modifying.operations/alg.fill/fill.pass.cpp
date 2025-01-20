//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, class T>
//   requires OutputIterator<Iter, const T&>
//   constexpr void      // constexpr after C++17
//   fill(Iter first, Iter last, const T& value);

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter, class Container>
TEST_CONSTEXPR_CXX20 void
test(Container in, size_t from, size_t to, typename Container::value_type value, Container expected) {
  std::fill(Iter(in.data() + from), Iter(in.data() + to), value);
  assert(in == expected);
}

template <class T>
struct Test {
  template <class Iter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    {
      std::array<T, 4> in       = {1, 2, 3, 4};
      std::array<T, 4> expected = {5, 5, 5, 5};
      test<Iter>(in, 0, 4, 5, expected);
    }
    {
      std::array<T, 4> in       = {1, 2, 3, 4};
      std::array<T, 4> expected = {1, 5, 5, 4};
      test<Iter>(in, 1, 3, 5, expected);
    }
  }
};

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::forward_iterator_list<char*>(), Test<char>());
  types::for_each(types::forward_iterator_list<int*>(), Test<int>());
  {   // test vector<bool>::iterator optimization
    { // simple case
      std::vector<bool> in(4, false);
      std::vector<bool> expected(4, true);
      std::fill(in.begin(), in.end(), true);
      assert(in == expected);
    }
    { // partial byte in the front is not filled
      std::vector<bool> in(8, false);
      std::vector<bool> expected(8, true);
      expected[0] = false;
      expected[1] = false;
      std::fill(in.begin() + 2, in.end(), true);
      assert(in == expected);
    }
    { // partial byte in the back is not filled
      std::vector<bool> in(8, false);
      std::vector<bool> expected(8, true);
      expected[6] = false;
      expected[7] = false;
      std::fill(in.begin(), in.end() - 2, true);
      assert(in == expected);
    }
    { // partial byte in the front and back is not filled
      std::vector<bool> in(16, false);
      std::vector<bool> expected(16, true);
      expected[0]  = false;
      expected[1]  = false;
      expected[14] = false;
      expected[15] = false;
      std::fill(in.begin() + 2, in.end() - 2, true);
      assert(in == expected);
    }
    { // only a few bits of a byte are set
      std::vector<bool> in(8, false);
      std::vector<bool> expected(8, true);
      expected[0] = false;
      expected[1] = false;
      expected[6] = false;
      expected[7] = false;
      std::fill(in.begin() + 2, in.end() - 2, true);
      assert(in == expected);
    }
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
