//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// friend constexpr iterator operator-(iterator i, difference_type n)
//   requires advanceable<W>;
// friend constexpr difference_type operator-(const iterator& x, const iterator& y)
//   requires advanceable<W>;

#include <cassert>
#include <cstdint>
#include <ranges>

#include "test_macros.h"
#include "../types.h"

// If we're compiling for 32 bit or windows, int and long are the same size, so long long is the correct difference type.
#if INTPTR_MAX == INT32_MAX || defined(_WIN32)
using IntDiffT = long long;
#else
using IntDiffT = long;
#endif

constexpr bool test() {
  // <iterator> - difference_type
  {
    // When "_Start" is signed integer like.
    {
      std::ranges::iota_view<int> io(0);
      auto iter1 = std::next(io.begin(), 10);
      auto iter2 = std::next(io.begin(), 10);
      assert(iter1 == iter2);
      assert(iter1 - 5 != iter2);
      assert(iter1 - 5 == std::ranges::prev(iter2, 5));

      static_assert(!std::is_reference_v<decltype(iter2 - 5)>);
    }

    // When "_Start" is not integer like.
    {
      std::ranges::iota_view io(SomeInt(0));
      auto iter1 = std::next(io.begin(), 10);
      auto iter2 = std::next(io.begin(), 10);
      assert(iter1 == iter2);
      assert(iter1 - 5 != iter2);
      assert(iter1 - 5 == std::ranges::prev(iter2, 5));

      static_assert(!std::is_reference_v<decltype(iter2 - 5)>);
    }

    // When "_Start" is unsigned integer like and n is greater than or equal to zero.
    {
      std::ranges::iota_view<unsigned> io(0);
      auto iter1 = std::next(io.begin(), 10);
      auto iter2 = std::next(io.begin(), 10);
      assert(iter1 == iter2);
      assert(iter1 - 5 != iter2);
      assert(iter1 - 5 == std::ranges::prev(iter2, 5));

      static_assert(!std::is_reference_v<decltype(iter2 - 5)>);
    }
    {
      std::ranges::iota_view<unsigned> io(0);
      auto iter1 = std::next(io.begin(), 10);
      auto iter2 = std::next(io.begin(), 10);
      assert(iter1 - 0 == iter2);
    }

    // When "_Start" is unsigned integer like and n is less than zero.
    {
      std::ranges::iota_view<unsigned> io(0);
      auto iter1 = std::next(io.begin(), 10);
      auto iter2 = std::next(io.begin(), 10);
      assert(iter1 - 5 != iter2);
      assert(iter1 - 5 == std::ranges::prev(iter2, 5));

      static_assert(!std::is_reference_v<decltype(iter2 - 5)>);
    }
  }

  // <iterator> - <iterator>
  {
    // When "_Start" is signed integer like.
    {
      std::ranges::iota_view<int> io(0);
      auto iter1 = std::next(io.begin(), 10);
      auto iter2 = std::next(io.begin(), 5);
      assert(iter1 - iter2 == 5);

      LIBCPP_STATIC_ASSERT(std::same_as<decltype(iter1 - iter2), IntDiffT>);
    }
    {
      std::ranges::iota_view<int> io(0);
      auto iter1 = std::next(io.begin(), 10);
      auto iter2 = std::next(io.begin(), 10);
      assert(iter1 - iter2 == 0);

      LIBCPP_STATIC_ASSERT(std::same_as<decltype(iter1 - iter2), IntDiffT>);
    }
    {
      std::ranges::iota_view<int> io(0);
      auto iter1 = std::next(io.begin(), 5);
      auto iter2 = std::next(io.begin(), 10);
      assert(iter1 - iter2 == -5);

      LIBCPP_STATIC_ASSERT(std::same_as<decltype(iter1 - iter2), IntDiffT>);
    }

    // When "_Start" is unsigned integer like and y > x.
    {
      std::ranges::iota_view<unsigned> io(0);
      auto iter1 = std::next(io.begin(), 5);
      auto iter2 = std::next(io.begin(), 10);
      assert(iter1 - iter2 == -5);

      LIBCPP_STATIC_ASSERT(std::same_as<decltype(iter1 - iter2), IntDiffT>);
    }

    // When "_Start" is unsigned integer like and x >= y.
    {
      std::ranges::iota_view<unsigned> io(0);
      auto iter1 = std::next(io.begin(), 10);
      auto iter2 = std::next(io.begin(), 5);
      assert(iter1 - iter2 == 5);

      LIBCPP_STATIC_ASSERT(std::same_as<decltype(iter1 - iter2), IntDiffT>);
    }
    {
      std::ranges::iota_view<unsigned> io(0);
      auto iter1 = std::next(io.begin(), 10);
      auto iter2 = std::next(io.begin(), 10);
      assert(iter1 - iter2 == 0);

      LIBCPP_STATIC_ASSERT(std::same_as<decltype(iter1 - iter2), IntDiffT>);
    }

    // When "_Start" is not integer like.
    {
      std::ranges::iota_view io(SomeInt(0));
      auto iter1 = std::next(io.begin(), 10);
      auto iter2 = std::next(io.begin(), 5);
      assert(iter1 - iter2 == 5);

      static_assert(std::same_as<decltype(iter1 - iter2), int>);
    }
    {
      std::ranges::iota_view io(SomeInt(0));
      auto iter1 = std::next(io.begin(), 10);
      auto iter2 = std::next(io.begin(), 10);
      assert(iter1 - iter2 == 0);

      static_assert(std::same_as<decltype(iter1 - iter2), int>);
    }
    {
      std::ranges::iota_view io(SomeInt(0));
      auto iter1 = std::next(io.begin(), 5);
      auto iter2 = std::next(io.begin(), 10);
      assert(iter1 - iter2 == -5);

      static_assert(std::same_as<decltype(iter1 - iter2), int>);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
