//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto begin();

#include <array>
#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

#include "test_iterators.h"

struct View : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

// Test that begin is not const
template <class T>
concept HasBegin = requires(T t) { t.begin(); };

static_assert(HasBegin<std::ranges::split_view<View, View>>);
static_assert(!HasBegin<const std::ranges::split_view<View, View>>);

template <template <class> class MakeIter>
constexpr void testOne() {
  constexpr auto make_subrange = []<class T, std::size_t N>(T(&buffer)[N]) {
    using Iter = MakeIter<T*>;
    using Sent = sentinel_wrapper<Iter>;
    return std::ranges::subrange<Iter, Sent>{Iter{buffer}, Sent{Iter{buffer + N}}};
  };

  using Iter  = MakeIter<int*>;
  using Sent  = sentinel_wrapper<Iter>;
  using Range = std::ranges::subrange<Iter, Sent>;

  // empty view
  {
    std::array<int, 0> a;
    Range range{Iter{a.data()}, Sent{Iter{a.data() + a.size()}}};
    std::ranges::split_view sv{std::move(range), 1};
    auto it         = sv.begin();
    auto firstRange = *it;
    assert(firstRange.begin() == range.begin());
    assert(firstRange.end() == range.end());
  }

  // empty pattern
  {
    int buffer[] = {1, 2, 3};
    auto range   = make_subrange(buffer);
    std::ranges::split_view sv{std::move(range), std::views::empty<int>};
    auto it         = sv.begin();
    auto firstRange = *it;
    assert(firstRange.begin() == range.begin());
    assert(firstRange.end() == std::next(range.begin()));
  }

  // empty view and empty pattern
  {
    std::array<int, 0> a;
    Range range{Iter{a.data()}, Sent{Iter{a.data() + a.size()}}};
    std::ranges::split_view sv{std::move(range), std::views::empty<int>};
    auto it         = sv.begin();
    auto firstRange = *it;
    assert(firstRange.begin() == range.begin());
    assert(firstRange.end() == range.end());
  }

  // pattern found at the beginning
  {
    int buffer[]  = {1, 2, 3};
    auto range    = make_subrange(buffer);
    int pattern[] = {1, 2};
    std::ranges::split_view sv{range, pattern};

    auto it         = sv.begin();
    auto firstRange = *it;
    assert(firstRange.begin() == range.begin());
    assert(firstRange.end() == range.begin());
  }

  // pattern found in the middle
  {
    int buffer[]  = {1, 2, 3, 4};
    auto range    = make_subrange(buffer);
    int pattern[] = {2, 3};
    std::ranges::split_view sv{range, pattern};

    auto it         = sv.begin();
    auto firstRange = *it;
    assert(firstRange.begin() == range.begin());
    assert(firstRange.end() == std::next(range.begin()));
  }

  // pattern found at the end
  {
    int buffer[]  = {1, 2, 3};
    auto range    = make_subrange(buffer);
    int pattern[] = {2, 3};
    std::ranges::split_view sv{range, pattern};

    auto it         = sv.begin();
    auto firstRange = *it;
    assert(firstRange.begin() == range.begin());
    assert(firstRange.end() == std::next(range.begin()));
  }

  // pattern not found
  {
    int buffer[]  = {1, 2, 3};
    auto range    = make_subrange(buffer);
    int pattern[] = {1, 3};
    std::ranges::split_view sv{range, pattern};

    auto it         = sv.begin();
    auto firstRange = *it;
    assert(firstRange.begin() == range.begin());
    assert(firstRange.end() == range.end());
  }

  // Make sure that we cache the result of begin() on subsequent calls
  {
    struct Foo {
      int& equalsCalledTimes;

      constexpr bool operator==(const Foo&) const {
        ++equalsCalledTimes;
        return true;
      }
    };

    int equalsCalledTimes = 0;
    Foo buffer[]          = {Foo{equalsCalledTimes}, Foo{equalsCalledTimes}};
    auto range            = make_subrange(buffer);

    std::ranges::split_view sv{range, Foo{equalsCalledTimes}};

    assert(equalsCalledTimes == 0);

    [[maybe_unused]] auto it1       = sv.begin();
    auto calledTimesAfterFirstBegin = equalsCalledTimes;
    assert(calledTimesAfterFirstBegin != 0);

    for (int i = 0; i < 10; ++i) {
      [[maybe_unused]] auto it2 = sv.begin();
      assert(equalsCalledTimes == calledTimesAfterFirstBegin);
    }
  }
}

constexpr bool test() {
  testOne<forward_iterator>();
  testOne<bidirectional_iterator>();
  testOne<random_access_iterator>();
  testOne<contiguous_iterator>();
  testOne<std::type_identity_t>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
