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
// friend constexpr iter_rvalue_reference_t<Iterator>
//   iter_move(const reverse_iterator& i) noexcept(see below);

#include <iterator>

#include <cassert>
#include <type_traits>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"

constexpr bool test() {
  // Can use `iter_move` with a regular array.
  {
    constexpr int N = 3;
    int a[N] = {0, 1, 2};

    std::reverse_iterator<int*> ri(a + N);
    static_assert(std::same_as<decltype(iter_move(ri)), int&&>);
    assert(iter_move(ri) == 2);

    ++ri;
    assert(iter_move(ri) == 1);
  }

  // Check that the `iter_move` customization point is being used.
  {
    constexpr int N = 3;
    int a[N] = {0, 1, 2};

    int iter_move_invocations = 0;
    adl::Iterator i = adl::Iterator::TrackMoves(a + N, iter_move_invocations);
    std::reverse_iterator<adl::Iterator> ri(i);
    int x = iter_move(ri);
    assert(x == 2);
    assert(iter_move_invocations == 1);
  }

  // Check the `noexcept` specification.
  {
    {
      struct ThrowingCopyNoexceptDecrement {
        using value_type = int;
        using difference_type = std::ptrdiff_t;

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
      ASSERT_NOEXCEPT(std::ranges::iter_move(--std::declval<ThrowingCopyNoexceptDecrement&>()));
      using RI = std::reverse_iterator<ThrowingCopyNoexceptDecrement>;
      ASSERT_NOT_NOEXCEPT(iter_move(std::declval<RI>()));
    }

    {
      struct NoexceptCopyThrowingDecrement {
        using value_type = int;
        using difference_type = std::ptrdiff_t;

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
      ASSERT_NOT_NOEXCEPT(std::ranges::iter_move(--std::declval<NoexceptCopyThrowingDecrement&>()));
      using RI = std::reverse_iterator<NoexceptCopyThrowingDecrement>;
      ASSERT_NOT_NOEXCEPT(iter_move(std::declval<RI>()));
    }

    {
      struct NoexceptCopyAndDecrement {
        using value_type = int;
        using difference_type = std::ptrdiff_t;

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
      ASSERT_NOEXCEPT(std::ranges::iter_move(--std::declval<NoexceptCopyAndDecrement&>()));
      using RI = std::reverse_iterator<NoexceptCopyAndDecrement>;
      ASSERT_NOEXCEPT(iter_move(std::declval<RI>()));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
