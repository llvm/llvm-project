//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// friend constexpr decltype(auto) iter_move(const inner-iterator& i)
//   noexcept(noexcept(ranges::iter_move(i.i_.<current>)));

#include <iterator>

#include <cassert>
#include <type_traits>
#include <utility>
#include "../types.h"

namespace adl {

template <bool IsNoexcept = false>
struct Iterator {
  using value_type = int;
  using difference_type = ptrdiff_t;

  value_type* ptr_ = nullptr;
  int* iter_move_invocations_ = nullptr;

  constexpr Iterator() = default;
  constexpr explicit Iterator(int* p, int& iter_moves) : ptr_(p), iter_move_invocations_(&iter_moves) {}

  constexpr value_type& operator*() const { return *ptr_; }

  Iterator& operator++() { ++ptr_; return *this; }
  Iterator operator++(int) {
    Iterator prev = *this;
    ++ptr_;
    return prev;
  }

  constexpr Iterator& operator--() { --ptr_; return *this; }
  constexpr Iterator operator--(int) {
    Iterator prev = *this;
    --ptr_;
    return prev;
  }

  constexpr friend value_type&& iter_move(Iterator iter) noexcept(IsNoexcept) {
    if (iter.iter_move_invocations_) {
      ++(*iter.iter_move_invocations_);
    }
    return std::move(*iter);
  }

  friend bool operator==(const Iterator& lhs, const Iterator& rhs) { return lhs.ptr_ == rhs.ptr_; }
};

template <bool IsNoexcept = false>
struct View : std::ranges::view_base {
  static constexpr int N = 3;
  int a[N] = {0, 1, 2};
  int* iter_moves = nullptr;

  constexpr View() = default;
  constexpr View(int& iter_move_invocations) : iter_moves(&iter_move_invocations) {
  }

  constexpr adl::Iterator<IsNoexcept> begin() { return adl::Iterator<IsNoexcept>(a, *iter_moves); }
  constexpr adl::Iterator<IsNoexcept> end() { return adl::Iterator<IsNoexcept>(a + N, *iter_moves); }
};

} // namespace adl

constexpr bool test() {
  // Can use `iter_move` with `inner-iterator`; `View` is a forward range.
  {
    SplitViewForward v("abc def", " ");
    auto segment = *v.begin();

    // Non-const iterator.
    {
      auto i = segment.begin();
      static_assert(std::same_as<decltype(iter_move(i)), const char &&>);
      assert(iter_move(i) == 'a');
    }

    // Const iterator.
    {
      const auto i = segment.begin();
      static_assert(std::same_as<decltype(iter_move(i)), const char &&>);
      assert(iter_move(i) == 'a');
    }
  }

  // Can use `iter_move` with `inner-iterator`, `View` is an input range.
  {
    SplitViewInput v("abc def", ' ');
    auto segment = *v.begin();

    // Non-const iterator.
    {
      auto i = segment.begin();
      static_assert(std::same_as<decltype(iter_move(i)), char &&>);
      assert(iter_move(i) == 'a');
    }

    // Const iterator.
    {
      const auto i = segment.begin();
      static_assert(std::same_as<decltype(iter_move(i)), char &&>);
      assert(iter_move(i) == 'a');
    }
  }

  // Ensure the `iter_move` customization point is being used.
  {
    int iter_move_invocations = 0;
    adl::View<> input(iter_move_invocations);
    std::ranges::lazy_split_view<adl::View<>, adl::View<>> v(input, adl::View<>());

    auto segment = *v.begin();
    auto i = segment.begin();
    int x = iter_move(i);
    assert(x == 0);
    assert(iter_move_invocations == 1);
  }

  // Check the `noexcept` specification.
  {
    {
      using ThrowingSplitView = std::ranges::lazy_split_view<adl::View<false>, adl::View<false>>;
      using ThrowingValueType = std::ranges::iterator_t<ThrowingSplitView>::value_type;
      using ThrowingIter = std::ranges::iterator_t<ThrowingValueType>;
      ASSERT_NOT_NOEXCEPT(std::ranges::iter_move(std::declval<adl::Iterator<false>>()));
      ASSERT_NOT_NOEXCEPT(iter_move(std::declval<ThrowingIter>()));
    }

    {
      using NoexceptSplitView = std::ranges::lazy_split_view<adl::View<true>, adl::View<true>>;
      using NoexceptValueType = std::ranges::iterator_t<NoexceptSplitView>::value_type;
      using NoexceptIter = std::ranges::iterator_t<NoexceptValueType>;
      ASSERT_NOEXCEPT(std::ranges::iter_move(std::declval<adl::Iterator<true>>()));
      ASSERT_NOEXCEPT(iter_move(std::declval<NoexceptIter>()));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
