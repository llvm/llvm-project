//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr decltype(auto) operator*() const noexcept(see below);

#include <array>
#include <cassert>
#include <ranges>

#include "../types.h"

// Test noexcept
// Remarks: Let Is be the pack 0, 1, ..., (sizeof...(Views)-1). The exception specification is equivalent to:
//   noexcept(invoke(*parent_->fun_, *std::get<Is>(inner_.current_)...)).

template <class ZipTransformView>
concept DerefNoexcept = requires(std::ranges::iterator_t<ZipTransformView> iter) { requires noexcept(*iter); };

struct ThrowingDerefIter {
  using iterator_category = std::forward_iterator_tag;
  using value_type        = int;
  using difference_type   = std::intptr_t;

  int operator*() const noexcept(false);

  ThrowingDerefIter& operator++();
  void operator++(int);

  friend constexpr bool operator==(const ThrowingDerefIter&, const ThrowingDerefIter&) = default;
};

using NoexceptDerefIter = int*;

template <bool NoExceptDeref>
struct TestView : std::ranges::view_base {
  using Iter = std::conditional_t<NoExceptDeref, NoexceptDerefIter, ThrowingDerefIter>;
  Iter begin() const;
  Iter end() const;
};

template <bool NoExceptCall>
struct TestFn {
  int operator()(auto&&...) const noexcept(NoExceptCall);
};

static_assert(DerefNoexcept<std::ranges::zip_transform_view<TestFn<true>, TestView<true>>>);
static_assert(DerefNoexcept<std::ranges::zip_transform_view<TestFn<true>, TestView<true>, TestView<true>>>);
static_assert(!DerefNoexcept<std::ranges::zip_transform_view<TestFn<true>, TestView<false>>>);
static_assert(!DerefNoexcept<std::ranges::zip_transform_view<TestFn<false>, TestView<true>>>);
static_assert(!DerefNoexcept<std::ranges::zip_transform_view<TestFn<false>, TestView<false>>>);
static_assert(!DerefNoexcept<std::ranges::zip_transform_view<TestFn<false>, TestView<false>, TestView<true>>>);
static_assert(!DerefNoexcept<std::ranges::zip_transform_view<TestFn<true>, TestView<false>, TestView<true>>>);
static_assert(!DerefNoexcept<std::ranges::zip_transform_view<TestFn<false>, TestView<false>, TestView<false>>>);

constexpr bool test() {
  std::array a{1, 2, 3, 4};
  std::array b{4.1, 3.2, 4.3};
  {
    // Function returns reference
    std::ranges::zip_transform_view v(GetFirst{}, a);
    auto it                               = v.begin();
    std::same_as<int&> decltype(auto) val = *it;
    assert(&val == &a[0]);
  }

  {
    // function returns PRValue
    std::ranges::zip_transform_view v(MakeTuple{}, a, b);
    auto it                                                  = v.begin();
    std::same_as<std::tuple<int, double>> decltype(auto) val = *it;
    assert(val == std::tuple(1, 4.1));
  }

  {
    // operator* is const
    std::ranges::zip_transform_view v(GetFirst{}, a);
    const auto it                         = v.begin();
    std::same_as<int&> decltype(auto) val = *it;
    assert(&val == &a[0]);
  }

  {
    // dereference twice
    std::ranges::zip_transform_view v(MakeTuple{}, a, b);
    auto it = v.begin();
    assert(*it == std::tuple(1, 4.1));
    assert(*it == std::tuple(1, 4.1));
  }

  {
    // back and forth
    std::ranges::zip_transform_view v(Tie{}, a, b);
    auto it = v.begin();
    assert(&std::get<0>(*it) == &a[0]);
    assert(&std::get<1>(*it) == &b[0]);
    ++it;
    assert(&std::get<0>(*it) == &a[1]);
    assert(&std::get<1>(*it) == &b[1]);
    --it;
    assert(&std::get<0>(*it) == &a[0]);
    assert(&std::get<1>(*it) == &b[0]);
  }

  int buffer[] = {1, 2, 3, 4, 5, 6};
  {
    // one range
    std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer});
    auto it = v.begin();
    assert(*it == std::make_tuple(1));
  }

  {
    // two ranges
    std::ranges::zip_transform_view v(GetFirst{}, SimpleCommon{buffer}, std::views::iota(0));
    auto it = v.begin();
    assert(&*it == &buffer[0]);
  }

  {
    // three ranges
    std::ranges::zip_transform_view v(Tie{}, SimpleCommon{buffer}, SimpleCommon{buffer}, std::ranges::single_view(2.));
    auto it = v.begin();
    assert(&std::get<0>(*it) == &buffer[0]);
    assert(&std::get<1>(*it) == &buffer[0]);
    assert(std::get<2>(*it) == 2.0);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
