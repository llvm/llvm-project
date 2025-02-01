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

TEST_CONSTEXPR_CXX20 bool test_vector_bool(std::size_t N) {
  { // Test with full bytes
    std::vector<bool> in(N, false);
    std::vector<bool> expected(N, true);
    std::fill(in.begin(), in.end(), true);
    assert(in == expected);
  }
  { // Test with partial bytes
    std::vector<bool> in(N, false);
    std::vector<bool> expected(N, true);
    std::fill(in.begin() + 4, in.end() - 4, true);
    assert(std::equal(in.begin() + 4, in.end() - 4, expected.begin()));
  }

  return true;
}

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::forward_iterator_list<char*>(), Test<char>());
  types::for_each(types::forward_iterator_list<int*>(), Test<int>());

  { // Test vector<bool>::iterator optimization
    assert(test_vector_bool(8));
    assert(test_vector_bool(19));
    assert(test_vector_bool(32));
    assert(test_vector_bool(49));
    assert(test_vector_bool(64));
    assert(test_vector_bool(199));
    assert(test_vector_bool(256));
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
