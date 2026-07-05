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

struct Pred {
  constexpr bool operator()(int i) const { return i < 3; }
};

static_assert(HasBegin<std::ranges::drop_while_view<View, Pred>>);
static_assert(!HasBegin<const std::ranges::drop_while_view<View, Pred>>);

constexpr auto always = [](auto v) { return [v](auto&&...) { return v; }; };

template <class Iter>
constexpr void testOne() {
  using Sent                   = sentinel_wrapper<Iter>;
  using Range                  = std::ranges::subrange<Iter, Sent>;
  constexpr auto make_subrange = []<std::size_t N>(int(&buffer)[N]) {
    return Range{Iter{buffer}, Sent{Iter{buffer + N}}};
  };

  // empty
  {
    std::array<int, 0> a;
    Range range{Iter{a.data()}, Sent{Iter{a.data() + a.size()}}};
    std::ranges::drop_while_view dwv{std::move(range), always(false)};
    std::same_as<Iter> decltype(auto) it = dwv.begin();
    assert(base(it) == a.data() + a.size());
  }

  // 1 element not dropped
  {
    int buffer[] = {1};
    auto range   = make_subrange(buffer);
    std::ranges::drop_while_view dwv{std::move(range), always(false)};
    std::same_as<Iter> decltype(auto) it = dwv.begin();
    assert(base(it) == buffer);
  }

  // 1 element dropped
  {
    int buffer[] = {1};
    auto range   = make_subrange(buffer);
    std::ranges::drop_while_view dwv{std::move(range), always(true)};
    std::same_as<Iter> decltype(auto) it = dwv.begin();
    assert(base(it) == buffer + 1);
  }

  // multiple elements. no element dropped
  {
    int buffer[] = {1, 2, 3, 4, 5};
    auto range   = make_subrange(buffer);
    std::ranges::drop_while_view dwv{std::move(range), always(false)};
    std::same_as<Iter> decltype(auto) it = dwv.begin();
    assert(base(it) == buffer);
  }

  // multiple elements. all elements dropped
  {
    int buffer[] = {1, 2, 3, 4, 5};
    auto range   = make_subrange(buffer);
    std::ranges::drop_while_view dwv{std::move(range), always(true)};
    std::same_as<Iter> decltype(auto) it = dwv.begin();
    assert(base(it) == buffer + 5);
  }

  // multiple elements. some elements dropped
  {
    int buffer[] = {1, 2, 3, 2, 1};
    auto range   = make_subrange(buffer);
    std::ranges::drop_while_view dwv{std::move(range), [](int i) { return i < 3; }};
    std::same_as<Iter> decltype(auto) it = dwv.begin();
    assert(base(it) == buffer + 2);
  }

  // Make sure we do not make a copy of the predicate when we call begin()
  {
    struct TrackingPred {
      constexpr explicit TrackingPred(bool* moved, bool* copied) : moved_(moved), copied_(copied) {}
      constexpr TrackingPred(TrackingPred const& other) : moved_(other.moved_), copied_(other.copied_) {
        *copied_ = true;
      }
      constexpr TrackingPred(TrackingPred&& other) : moved_(other.moved_), copied_(other.copied_) { *moved_ = true; }
      TrackingPred& operator=(TrackingPred const&) = default;
      TrackingPred& operator=(TrackingPred&&)      = default;

      constexpr bool operator()(int i) const { return i < 3; }
      bool* moved_;
      bool* copied_;
    };

    int buffer[] = {1, 2, 3, 2, 1};
    bool moved = false, copied = false;
    auto range = make_subrange(buffer);
    std::ranges::drop_while_view dwv{std::move(range), TrackingPred(&moved, &copied)};
    moved                    = false;
    copied                   = false;
    [[maybe_unused]] auto it = dwv.begin();
    assert(!moved);
    assert(!copied);
  }

  // Test with a non-const predicate
  {
    int buffer[] = {1, 2, 3, 2, 1};
    auto range   = make_subrange(buffer);
    std::ranges::drop_while_view dwv{std::move(range), [](int i) mutable { return i < 3; }};
    std::same_as<Iter> decltype(auto) it = dwv.begin();
    assert(base(it) == buffer + 2);
  }

  // Test with a predicate that takes by non-const reference
  {
    int buffer[] = {1, 2, 3, 2, 1};
    auto range   = make_subrange(buffer);
    std::ranges::drop_while_view dwv{std::move(range), [](int& i) { return i < 3; }};
    std::same_as<Iter> decltype(auto) it = dwv.begin();
    assert(base(it) == buffer + 2);
  }

  if constexpr (std::forward_iterator<Iter>) {
    // Make sure that we cache the result of begin() on subsequent calls
    {
      int buffer[] = {1, 2, 3, 2, 1};
      auto range   = make_subrange(buffer);

      int called = 0;
      auto pred  = [&](int i) {
        ++called;
        return i < 3;
      };
      std::ranges::drop_while_view dwv{range, pred};
      for (auto i = 0; i < 10; ++i) {
        std::same_as<Iter> decltype(auto) it = dwv.begin();
        assert(base(it) == buffer + 2);
        assert(called == 3);
      }
    }
  }
}

constexpr bool test() {
  testOne<cpp17_input_iterator<int*>>();
  testOne<cpp20_input_iterator<int*>>();
  testOne<forward_iterator<int*>>();
  testOne<bidirectional_iterator<int*>>();
  testOne<random_access_iterator<int*>>();
  testOne<contiguous_iterator<int*>>();
  testOne<int*>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
