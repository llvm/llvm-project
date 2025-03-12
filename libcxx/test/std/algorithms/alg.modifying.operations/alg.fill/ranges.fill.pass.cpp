//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T, output_iterator<const T&> O, sentinel_for<O> S>
//   constexpr O ranges::fill(O first, S last, const T& value);
// template<class T, output_range<const T&> R>
//   constexpr borrowed_iterator_t<R> ranges::fill(R&& r, const T& value);

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>
#include <string>
#include <vector>

#include "sized_allocator.h"
#include "almost_satisfies_types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter, class Sent = sentinel_wrapper<Iter>>
concept HasFillIt = requires(Iter iter, Sent sent) { std::ranges::fill(iter, sent, int{}); };

static_assert(HasFillIt<int*>);
static_assert(!HasFillIt<OutputIteratorNotIndirectlyWritable>);
static_assert(!HasFillIt<OutputIteratorNotInputOrOutputIterator>);
static_assert(!HasFillIt<int*, SentinelForNotSemiregular>);
static_assert(!HasFillIt<int*, SentinelForNotWeaklyEqualityComparableWith>);

template <class Range>
concept HasFillR = requires(Range range) { std::ranges::fill(range, int{}); };

static_assert(HasFillR<UncheckedRange<int*>>);
static_assert(!HasFillR<OutputRangeNotIndirectlyWritable>);
static_assert(!HasFillR<OutputRangeNotInputOrOutputIterator>);
static_assert(!HasFillR<OutputRangeNotSentinelSemiregular>);
static_assert(!HasFillR<OutputRangeNotSentinelEqualityComparableWith>);

template <class It, class Sent = It>
constexpr void test_iterators() {
  { // simple test
    {
      int a[3];
      std::same_as<It> auto ret = std::ranges::fill(It(a), Sent(It(a + 3)), 1);
      assert(std::all_of(a, a + 3, [](int i) { return i == 1; }));
      assert(base(ret) == a + 3);
    }
    {
      int a[3];
      auto range                = std::ranges::subrange(It(a), Sent(It(a + 3)));
      std::same_as<It> auto ret = std::ranges::fill(range, 1);
      assert(std::all_of(a, a + 3, [](int i) { return i == 1; }));
      assert(base(ret) == a + 3);
    }
  }

  { // check that an empty range works
    {
      std::array<int, 0> a;
      auto ret = std::ranges::fill(It(a.data()), Sent(It(a.data())), 1);
      assert(base(ret) == a.data());
    }
    {
      std::array<int, 0> a;
      auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data())));
      auto ret   = std::ranges::fill(range, 1);
      assert(base(ret) == a.data());
    }
  }
}

// Make sure std::ranges::fill behaves properly with std::vector<bool> iterators with custom
// size types. See https://github.com/llvm/llvm-project/pull/122410.
//
// The `ranges::{fill, fill_n}` algorithms require `vector<bool, Alloc>::iterator` to satisfy
// the `std::indirectly_writable` concept when used with `vector<bool, Alloc>`, which is only
// satisfied since C++23.
#if TEST_STD_VER >= 23
TEST_CONSTEXPR_CXX20 void test_bititer_with_custom_sized_types() {
  {
    using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
    std::vector<bool, Alloc> in(100, false, Alloc(1));
    std::vector<bool, Alloc> expected(100, true, Alloc(1));
    std::ranges::fill(in, true);
    assert(in == expected);
  }
  {
    using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
    std::vector<bool, Alloc> in(200, false, Alloc(1));
    std::vector<bool, Alloc> expected(200, true, Alloc(1));
    std::ranges::fill(in, true);
    assert(in == expected);
  }
  {
    using Alloc = sized_allocator<bool, std::uint32_t, std::int32_t>;
    std::vector<bool, Alloc> in(200, false, Alloc(1));
    std::vector<bool, Alloc> expected(200, true, Alloc(1));
    std::ranges::fill(in, true);
    assert(in == expected);
  }
  {
    using Alloc = sized_allocator<bool, std::uint64_t, std::int64_t>;
    std::vector<bool, Alloc> in(200, false, Alloc(1));
    std::vector<bool, Alloc> expected(200, true, Alloc(1));
    std::ranges::fill(in, true);
    assert(in == expected);
  }
}
#endif

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
        copied = true;
        return *this;
      }
    };
    {
      S a[5];
      std::ranges::fill(a, a + 5, S{true});
      assert(std::all_of(a, a + 5, [](S& s) { return s.copied; }));
    }
    {
      S a[5];
      std::ranges::fill(a, S{true});
      assert(std::all_of(a, a + 5, [](S& s) { return s.copied; }));
    }
  }

  { // check that std::ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> decltype(auto) ret =
        std::ranges::fill(std::array<int, 10>{}, 1);
  }

  { // check that std::ranges::dangling isn't returned with a borrowing range
    std::array<int, 10> a{};
    [[maybe_unused]] std::same_as<std::array<int, 10>::iterator> decltype(auto) ret =
        std::ranges::fill(std::views::all(a), 1);
    assert(std::all_of(a.begin(), a.end(), [](int i) { return i == 1; }));
  }

  { // check that non-trivially copyable items are copied properly
    {
      std::array<std::string, 10> a;
      auto ret = std::ranges::fill(a.begin(), a.end(), "long long string so no SSO");
      assert(ret == a.end());
      assert(std::all_of(a.begin(), a.end(), [](auto& s) { return s == "long long string so no SSO"; }));
    }
    {
      std::array<std::string, 10> a;
      auto ret = std::ranges::fill(a, "long long string so no SSO");
      assert(ret == a.end());
      assert(std::all_of(a.begin(), a.end(), [](auto& s) { return s == "long long string so no SSO"; }));
    }
  }

#if TEST_STD_VER >= 23
  test_bititer_with_custom_sized_types();
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
