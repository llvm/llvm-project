//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T, output_iterator<const T&> O>
//   constexpr O ranges::fill_n(O first, iter_difference_t<O> n, const T& value);

#include <algorithm>
#include <array>
#include <cassert>
#include <deque>
#include <ranges>
#include <string>
#include <vector>

#include "sized_allocator.h"
#include "almost_satisfies_types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
concept HasFillN = requires(Iter iter) { std::ranges::fill_n(iter, int{}, int{}); };

struct WrongType {};

static_assert(HasFillN<int*>);
static_assert(!HasFillN<WrongType*>);
static_assert(!HasFillN<OutputIteratorNotIndirectlyWritable>);
static_assert(!HasFillN<OutputIteratorNotInputOrOutputIterator>);

template <class It, class Sent = It>
constexpr void test_iterators() {
  { // simple test
    int a[3];
    std::same_as<It> decltype(auto) ret = std::ranges::fill_n(It(a), 3, 1);
    assert(std::all_of(a, a + 3, [](int i) { return i == 1; }));
    assert(base(ret) == a + 3);
  }

  { // check that an empty range works
    std::array<int, 0> a;
    auto ret = std::ranges::fill_n(It(a.data()), 0, 1);
    assert(base(ret) == a.data());
  }
}

// The `ranges::{fill, fill_n}` algorithms require `vector<bool, Alloc>::iterator` to satisfy
// the `std::indirectly_writable` concept when used with `vector<bool, Alloc>`, which is only
// satisfied since C++23.
#if TEST_STD_VER >= 23
constexpr bool test_vector_bool(std::size_t N) {
  {   // Test cases validating leading/trailing bits unfilled remain unchanged
    { // Leading bits are not filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      expected[0] = expected[1] = false;
      std::ranges::fill_n(std::ranges::begin(in) + 2, N - 2, true);
      assert(in == expected);
    }
    { // Trailing bits are not filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      expected[N - 1] = expected[N - 2] = false;
      std::ranges::fill_n(std::ranges::begin(in), N - 2, true);
      assert(in == expected);
    }
    { // Leading and trailing bits are not filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      expected[0] = expected[1] = expected[N - 1] = expected[N - 2] = false;
      std::ranges::fill_n(std::ranges::begin(in) + 2, N - 4, true);
      assert(in == expected);
    }
  }

  {   // Test cases with full or partial bytes filled
    { // Full bytes filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      std::ranges::fill_n(std::ranges::begin(in), N, true);
      assert(in == expected);
    }
    { // Partial bytes with offset filled
      std::vector<bool> in(N, false);
      std::vector<bool> expected(N, true);
      std::ranges::fill_n(std::ranges::begin(in) + 4, N - 8, true);
      std::ranges::fill_n(std::ranges::begin(expected), 4, false);
      std::ranges::fill_n(std::ranges::end(expected) - 4, 4, false);
      assert(in == expected);
    }
  }

  return true;
}
#endif

/*TEST_CONSTEXPR_CXX26*/ void test_deque() { // TODO: Mark as TEST_CONSTEXPR_CXX26 once std::deque is constexpr
  std::deque<int> in(20);
  std::deque<int> expected(in.size(), 42);
  std::ranges::fill_n(std::ranges::begin(in), std::ranges::size(in), 42);
  assert(in == expected);
}

constexpr bool test() {
  test_iterators<cpp17_output_iterator<int*>, sentinel_wrapper<cpp17_output_iterator<int*>>>();
  test_iterators<cpp20_output_iterator<int*>, sentinel_wrapper<cpp20_output_iterator<int*>>>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();
  test_iterators<int*>();

  { // check that every element is copied once
    struct S {
      bool copied = false;
      constexpr S& operator=(const S&) {
        assert(!copied);
        copied = true;
        return *this;
      }
    };

    S a[5];
    std::ranges::fill_n(a, 5, S{});
    assert(std::all_of(a, a + 5, [](S& s) { return s.copied; }));
  }

  { // check that non-trivially copyable items are copied properly
    std::array<std::string, 10> a;
    auto ret = std::ranges::fill_n(a.data(), 10, "long long string so no SSO");
    assert(ret == a.data() + a.size());
    assert(std::all_of(a.begin(), a.end(), [](auto& s) { return s == "long long string so no SSO"; }));
  }

#if TEST_STD_VER >= 23
  { // Test vector<bool>::iterator optimization
    assert(test_vector_bool(8));
    assert(test_vector_bool(19));
    assert(test_vector_bool(32));
    assert(test_vector_bool(49));
    assert(test_vector_bool(64));
    assert(test_vector_bool(199));
    assert(test_vector_bool(256));

    // Make sure std::ranges::fill_n behaves properly with std::vector<bool> iterators with custom
    // size types. See https://github.com/llvm/llvm-project/pull/122410.
    {
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(100, false, Alloc(1));
      std::vector<bool, Alloc> expected(100, true, Alloc(1));
      std::ranges::fill_n(std::ranges::begin(in), in.size(), true);
      assert(in == expected);
    }
    {
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(200, false, Alloc(1));
      std::vector<bool, Alloc> expected(200, true, Alloc(1));
      std::ranges::fill_n(std::ranges::begin(in), in.size(), true);
      assert(in == expected);
    }
    {
      using Alloc = sized_allocator<bool, std::uint32_t, std::int32_t>;
      std::vector<bool, Alloc> in(200, false, Alloc(1));
      std::vector<bool, Alloc> expected(200, true, Alloc(1));
      std::ranges::fill_n(std::ranges::begin(in), in.size(), true);
      assert(in == expected);
    }
    {
      using Alloc = sized_allocator<bool, std::uint64_t, std::int64_t>;
      std::vector<bool, Alloc> in(200, false, Alloc(1));
      std::vector<bool, Alloc> expected(200, true, Alloc(1));
      std::ranges::fill_n(std::ranges::begin(in), in.size(), true);
      assert(in == expected);
    }
  }
#endif

  if (!TEST_IS_CONSTANT_EVALUATED) // TODO: Use TEST_STD_AT_LEAST_26_OR_RUNTIME_EVALUATED when std::deque is made constexpr
    test_deque();

#if TEST_STD_VER >= 20
  {
    std::vector<std::vector<int>> v{{1, 2}, {1, 2, 3}, {}, {3, 4, 5}, {6}, {7, 8, 9, 6}, {0, 1, 2, 3, 0, 1, 2}};
    auto jv = std::ranges::join_view(v);
    std::ranges::fill_n(std::ranges::begin(jv), std::ranges::distance(jv), 42);
    for (const auto& vec : v)
      for (auto n : vec)
        assert(n == 42);
  }
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
