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
#include <deque>
#include <ranges>
#include <vector>

#include "sized_allocator.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"

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
  {   // Test cases validating leading/trailing bits unfilled remain unchanged
    { // Leading bits are not filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      expected[0] = expected[1] = false;
      std::fill(in.begin() + 2, in.end(), true);
      assert(in == expected);
    }
    { // Trailing bits are not filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      expected[N - 1] = expected[N - 2] = false;
      std::fill(in.begin(), in.end() - 2, true);
      assert(in == expected);
    }
    { // Leading and trailing bits are not filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      expected[0] = expected[1] = expected[N - 1] = expected[N - 2] = false;
      std::fill(in.begin() + 2, in.end() - 2, true);
      assert(in == expected);
    }
  }

  {   // Test cases with full or partial bytes filled
    { // Full bytes filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      std::fill(in.begin(), in.end(), true);
      assert(in == expected);
    }
    { // Partial bytes with offset filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      std::fill(in.begin() + 4, in.end() - 4, true);
      std::fill(expected.begin(), expected.begin() + 4, false);
      std::fill(expected.end() - 4, expected.end(), false);
      assert(in == expected);
    }
  }

  return true;
}

/*TEST_CONSTEXPR_CXX26*/ void test_deque() { // TODO: Mark as TEST_CONSTEXPR_CXX26 once std::deque is constexpr
  std::deque<int> in(20);
  std::deque<int> expected(in.size(), 42);
  std::fill(in.begin(), in.end(), 42);
  assert(in == expected);
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

    // Make sure std::fill behaves properly with std::vector<bool> iterators with custom size types.
    // See https://github.com/llvm/llvm-project/pull/122410.
    {
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(100, false, Alloc(1));
      std::vector<bool, Alloc> expected(100, true, Alloc(1));
      std::fill(in.begin(), in.end(), true);
      assert(in == expected);
    }
    {
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(200, false, Alloc(1));
      std::vector<bool, Alloc> expected(200, true, Alloc(1));
      std::fill(in.begin(), in.end(), true);
      assert(in == expected);
    }
    {
      using Alloc = sized_allocator<bool, std::uint32_t, std::int32_t>;
      std::vector<bool, Alloc> in(200, false, Alloc(1));
      std::vector<bool, Alloc> expected(200, true, Alloc(1));
      std::fill(in.begin(), in.end(), true);
      assert(in == expected);
    }
    {
      using Alloc = sized_allocator<bool, std::uint64_t, std::int64_t>;
      std::vector<bool, Alloc> in(200, false, Alloc(1));
      std::vector<bool, Alloc> expected(200, true, Alloc(1));
      std::fill(in.begin(), in.end(), true);
      assert(in == expected);
    }
  }

  if (!TEST_IS_CONSTANT_EVALUATED) // TODO: Use TEST_STD_AT_LEAST_26_OR_RUNTIME_EVALUATED when std::deque is made constexpr
    test_deque();

#if TEST_STD_VER >= 20
  { // Verify that join_view of vectors work properly.
    std::vector<std::vector<int>> v{{1, 2}, {1, 2, 3}, {}, {3, 4, 5}, {6}, {7, 8, 9, 6}, {0, 1, 2, 3, 0, 1, 2}};
    auto jv = std::ranges::join_view(v);
    std::fill(jv.begin(), jv.end(), 42);
    for (const auto& vec : v)
      for (auto n : vec)
        assert(n == 42);
  }
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
