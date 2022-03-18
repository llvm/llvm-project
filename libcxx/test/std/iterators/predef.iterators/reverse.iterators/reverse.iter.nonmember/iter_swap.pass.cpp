//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>
//
// reverse_iterator
//
// template<indirectly_swappable<Iterator> Iterator2>
//   friend constexpr void
//     iter_swap(const reverse_iterator& x,
//               const reverse_iterator<Iterator2>& y) noexcept(see below);

#include <iterator>

#include <cassert>
#include <type_traits>
#include <utility>
#include "test_macros.h"

namespace adl {

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

  constexpr friend void iter_swap(Iterator a, Iterator) {
    if (a.iter_swap_invocations_) {
      ++(*a.iter_swap_invocations_);
    }
  }

  friend bool operator==(const Iterator& lhs, const Iterator& rhs) { return lhs.ptr_ == rhs.ptr_; }
};

} // namespace adl

constexpr bool test() {
  // Can use `iter_swap` with a regular array.
  {
    constexpr int N = 3;
    int a[N] = {0, 1, 2};

    std::reverse_iterator rb(a + N);
    std::reverse_iterator re(a + 1);
    assert(a[0] == 0);
    assert(a[2] == 2);

    static_assert(std::same_as<decltype(iter_swap(rb, re)), void>);
    iter_swap(rb, re);
    assert(a[0] == 2);
    assert(a[2] == 0);
  }

  // Ensure the `iter_swap` customization point is being used.
  {
    int iter_swap_invocations = 0;
    adl::Iterator i1(iter_swap_invocations), i2(iter_swap_invocations);
    std::reverse_iterator<adl::Iterator> ri1(i1), ri2(i2);
    iter_swap(i1, i2);
    assert(iter_swap_invocations == 1);

    iter_swap(i2, i1);
    assert(iter_swap_invocations == 2);
  }

  // Check the `noexcept` specification.
  {
    {
      struct ThrowingCopyNoexceptDecrement {
        using value_type = int;
        using difference_type = ptrdiff_t;

        ThrowingCopyNoexceptDecrement();
        ThrowingCopyNoexceptDecrement(const ThrowingCopyNoexceptDecrement&);

        int& operator*() const noexcept { static int x; return x; }

        ThrowingCopyNoexceptDecrement& operator++();
        ThrowingCopyNoexceptDecrement operator++(int);
        ThrowingCopyNoexceptDecrement& operator--() noexcept;
        ThrowingCopyNoexceptDecrement operator--(int) noexcept;

        bool operator==(const ThrowingCopyNoexceptDecrement&) const = default;
      };
      static_assert(std::bidirectional_iterator<ThrowingCopyNoexceptDecrement>);

      static_assert(!std::is_nothrow_copy_constructible_v<ThrowingCopyNoexceptDecrement>);
      static_assert( std::is_nothrow_copy_constructible_v<int*>);
      ASSERT_NOEXCEPT(std::ranges::iter_swap(--std::declval<ThrowingCopyNoexceptDecrement&>(), --std::declval<int*&>()));
      using RI1 = std::reverse_iterator<ThrowingCopyNoexceptDecrement>;
      using RI2 = std::reverse_iterator<int*>;
      ASSERT_NOT_NOEXCEPT(iter_swap(std::declval<RI1>(), std::declval<RI2>()));
      ASSERT_NOT_NOEXCEPT(iter_swap(std::declval<RI2>(), std::declval<RI1>()));
    }

    {
      struct NoexceptCopyThrowingDecrement {
        using value_type = int;
        using difference_type = ptrdiff_t;

        NoexceptCopyThrowingDecrement();
        NoexceptCopyThrowingDecrement(const NoexceptCopyThrowingDecrement&) noexcept;

        int& operator*() const { static int x; return x; }

        NoexceptCopyThrowingDecrement& operator++();
        NoexceptCopyThrowingDecrement operator++(int);
        NoexceptCopyThrowingDecrement& operator--();
        NoexceptCopyThrowingDecrement operator--(int);

        bool operator==(const NoexceptCopyThrowingDecrement&) const = default;
      };
      static_assert(std::bidirectional_iterator<NoexceptCopyThrowingDecrement>);

      static_assert( std::is_nothrow_copy_constructible_v<NoexceptCopyThrowingDecrement>);
      static_assert( std::is_nothrow_copy_constructible_v<int*>);
      ASSERT_NOT_NOEXCEPT(std::ranges::iter_swap(--std::declval<NoexceptCopyThrowingDecrement&>(), --std::declval<int*&>()));
      using RI1 = std::reverse_iterator<NoexceptCopyThrowingDecrement>;
      using RI2 = std::reverse_iterator<int*>;
      ASSERT_NOT_NOEXCEPT(iter_swap(std::declval<RI1>(), std::declval<RI2>()));
      ASSERT_NOT_NOEXCEPT(iter_swap(std::declval<RI2>(), std::declval<RI1>()));
    }

    {
      struct NoexceptCopyAndDecrement {
        using value_type = int;
        using difference_type = ptrdiff_t;

        NoexceptCopyAndDecrement();
        NoexceptCopyAndDecrement(const NoexceptCopyAndDecrement&) noexcept;

        int& operator*() const noexcept { static int x; return x; }

        NoexceptCopyAndDecrement& operator++();
        NoexceptCopyAndDecrement operator++(int);
        NoexceptCopyAndDecrement& operator--() noexcept;
        NoexceptCopyAndDecrement operator--(int) noexcept;

        bool operator==(const NoexceptCopyAndDecrement&) const = default;
      };
      static_assert(std::bidirectional_iterator<NoexceptCopyAndDecrement>);

      static_assert( std::is_nothrow_copy_constructible_v<NoexceptCopyAndDecrement>);
      static_assert( std::is_nothrow_copy_constructible_v<int*>);
      ASSERT_NOEXCEPT(std::ranges::iter_swap(--std::declval<NoexceptCopyAndDecrement&>(), --std::declval<int*&>()));
      using RI1 = std::reverse_iterator<NoexceptCopyAndDecrement>;
      using RI2 = std::reverse_iterator<int*>;
      ASSERT_NOEXCEPT(iter_swap(std::declval<RI1>(), std::declval<RI2>()));
      ASSERT_NOEXCEPT(iter_swap(std::declval<RI2>(), std::declval<RI1>()));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
