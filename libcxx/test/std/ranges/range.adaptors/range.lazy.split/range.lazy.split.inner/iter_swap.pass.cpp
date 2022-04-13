//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// friend constexpr void iter_swap(const inner-iterator& x, const inner-iterator& y)
//   noexcept(noexcept(ranges::iter_swap(x.i_.<current>, y.i_.<current>)))
//   requires indirectly_swappable<iterator_t<Base>>;

#include <ranges>

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
  int* iter_swap_invocations_ = nullptr;

  constexpr Iterator() = default;
  constexpr explicit Iterator(int& iter_swaps) : iter_swap_invocations_(&iter_swaps) {}

  value_type& operator*() const { return *ptr_; }

  Iterator& operator++() { ++ptr_; return *this; }
  Iterator operator++(int) {
    Iterator prev = *this;
    ++ptr_;
    return prev;
  }

  Iterator& operator--() { --ptr_; return *this; }
  Iterator operator--(int) {
    Iterator prev = *this;
    --ptr_;
    return prev;
  }

  constexpr friend void iter_swap(Iterator a, Iterator) noexcept(IsNoexcept) {
    if (a.iter_swap_invocations_) {
      ++(*a.iter_swap_invocations_);
    }
  }

  friend bool operator==(const Iterator& lhs, const Iterator& rhs) { return lhs.ptr_ == rhs.ptr_; }
};

template <bool IsNoexcept = false>
struct View : std::ranges::view_base {
  int* iter_swaps = nullptr;

  constexpr View() = default;
  constexpr View(int& iter_swap_invocations) : iter_swaps(&iter_swap_invocations) {
  }

  constexpr adl::Iterator<IsNoexcept> begin() { return adl::Iterator<IsNoexcept>(*iter_swaps); }
  constexpr adl::Iterator<IsNoexcept> end() { return adl::Iterator<IsNoexcept>(*iter_swaps); }
};

} // namespace adl

constexpr bool test() {
  // Can use `iter_swap` with `inner-iterator`; `View` is a forward range.
  {
    // Non-const iterator.
    {
      SplitViewDiff v("abc def", " ");
      auto segment = *v.begin();

      auto i1 = segment.begin();
      auto i2 = i1++;
      static_assert(std::is_void_v<decltype(iter_swap(i1, i2))>);
      assert(*i1 == 'b');
      assert(*i2 == 'a');

      iter_swap(i1, i2);
      assert(*i1 == 'a');
      assert(*i2 == 'b');
      // Note that `iter_swap` swaps characters in the actual underlying range.
      assert(*v.base().begin() == 'b');
    }

    // Const iterator.
    {
      SplitViewDiff v("abc def", " ");
      auto segment = *v.begin();

      auto i1 = segment.begin();
      const auto i2 = i1++;
      static_assert(std::is_void_v<decltype(iter_swap(i1, i2))>);
      static_assert(std::is_void_v<decltype(iter_swap(i2, i2))>);
      assert(*i1 == 'b');
      assert(*i2 == 'a');

      iter_swap(i1, i2);
      assert(*i1 == 'a');
      assert(*i2 == 'b');
      assert(*v.base().begin() == 'b');
    }
  }

  // Can use `iter_swap` with `inner-iterator`; `View` is an input range.
  {

    // Non-const iterator.
    {
      // Iterators belong to the same view.
      {
        SplitViewInput v("abc def", ' ');
        auto segment = *v.begin();

        auto i1 = segment.begin();
        auto i2 = i1;
        ++i1;
        static_assert(std::is_void_v<decltype(iter_swap(i1, i2))>);
        assert(*i1 == 'b');
        // For an input view, all inner iterators are essentially thin proxies to the same underlying iterator.
        assert(*i2 == 'b');

        iter_swap(i1, i2);
        assert(*i1 == 'b');
        assert(*i2 == 'b');
      }

      // Iterators belong to different views.
      {
        SplitViewInput v1("abc def", ' ');
        auto val1 = *v1.begin();
        SplitViewInput v2 = v1;
        auto val2 = *v2.begin();

        auto i1 = val1.begin();
        auto i2 = val2.begin();
        ++i1;
        assert(*i1 == 'b');
        assert(*i2 == 'a');

        iter_swap(i1, i2);
        assert(*i1 == 'a');
        assert(*i2 == 'b');
      }
    }

    // Const iterator.
    {
      SplitViewInput v("abc def", ' ');
      auto segment = *v.begin();

      const auto i1 = segment.begin();
      const auto i2 = i1;
      static_assert(std::is_void_v<decltype(iter_swap(i1, i2))>);
      assert(*i1 == 'a');
      assert(*i2 == 'a');

      iter_swap(i1, i2);
      assert(*i1 == 'a');
      assert(*i2 == 'a');
    }
  }

  // Ensure the `iter_swap` customization point is being used.
  {
    int iter_swap_invocations = 0;
    adl::View<> input(iter_swap_invocations);
    std::ranges::lazy_split_view<adl::View<>, adl::View<>> v(input, adl::View<>());

    auto segment = *v.begin();
    auto i = segment.begin();
    iter_swap(i, i);
    assert(iter_swap_invocations == 1);
  }

  // Check the `noexcept` specification.
  {
    {
      using ThrowingSplitView = std::ranges::lazy_split_view<adl::View<false>, adl::View<false>>;
      using ThrowingValueType = std::ranges::iterator_t<ThrowingSplitView>::value_type;
      using ThrowingIter = std::ranges::iterator_t<ThrowingValueType>;
      ASSERT_NOT_NOEXCEPT(
          std::ranges::iter_swap(std::declval<adl::Iterator<false>>(), std::declval<adl::Iterator<false>>()));
      ASSERT_NOT_NOEXCEPT(iter_swap(std::declval<ThrowingIter>(), std::declval<ThrowingIter>()));
    }

    {
      using NoexceptSplitView = std::ranges::lazy_split_view<adl::View<true>, adl::View<true>>;
      using NoexceptValueType = std::ranges::iterator_t<NoexceptSplitView>::value_type;
      using NoexceptIter = std::ranges::iterator_t<NoexceptValueType>;
      ASSERT_NOEXCEPT(
          std::ranges::iter_swap(std::declval<adl::Iterator<true>>(), std::declval<adl::Iterator<true>>()));
      ASSERT_NOEXCEPT(iter_swap(std::declval<NoexceptIter>(), std::declval<NoexceptIter>()));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
