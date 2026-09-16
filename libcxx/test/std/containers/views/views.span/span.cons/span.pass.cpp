//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <span>

// template<class OtherElementType, size_t OtherExtent>
//   constexpr explicit(see below) span(const span<OtherElementType, OtherExtent>& s) noexcept;
//
// Constraints:
//   - extent == dynamic_extent || OtherExtent == dynamic_extent || extent == OtherExtent is true, and
//   - is_convertible_v<OtherElementType(*)[], element_type(*)[]> is true.

#include <span>
#include <cassert>
#include <string>

#include "test_macros.h"

template <class T, class From>
TEST_CONSTEXPR_CXX20 void check() {
  // dynamic -> dynamic
  {
    {
      std::span<From> from;
      std::span<T> span{from};
      ASSERT_NOEXCEPT(std::span<T>(from));
      assert(span.data() == nullptr);
      assert(span.size() == 0);
    }
    {
      From array[3] = {};
      std::span<From> from(array);
      std::span<T> span{from};
      ASSERT_NOEXCEPT(std::span<T>(from));
      assert(span.data() == array);
      assert(span.size() == 3);
    }
  }

  // static -> static
  {
    {
      std::span<From, 0> from;
      std::span<T, 0> span{from};
      ASSERT_NOEXCEPT(std::span<T, 0>(from));
      assert(span.data() == nullptr);
      assert(span.size() == 0);
    }

    {
      From array[3] = {};
      std::span<From, 3> from(array);
      std::span<T, 3> span{from};
      ASSERT_NOEXCEPT(std::span<T, 3>(from));
      assert(span.data() == array);
      assert(span.size() == 3);
    }
  }

  // static -> dynamic
  {
    {
      std::span<From, 0> from;
      std::span<T> span{from};
      ASSERT_NOEXCEPT(std::span<T>(from));
      assert(span.data() == nullptr);
      assert(span.size() == 0);
    }

    {
      From array[3] = {};
      std::span<From, 3> from(array);
      std::span<T> span{from};
      ASSERT_NOEXCEPT(std::span<T>(from));
      assert(span.data() == array);
      assert(span.size() == 3);
    }
  }

  // dynamic -> static (not allowed)
}

template <class T>
TEST_CONSTEXPR_CXX20 void check_cvs() {
  check<T, T>();

  check<T const, T>();
  check<T const, T const>();

  check<T volatile, T>();
  check<T volatile, T volatile>();

  check<T const volatile, T>();
  check<T const volatile, T const>();
  check<T const volatile, T volatile>();
  check<T const volatile, T const volatile>();
}

struct A {};

TEST_CONSTEXPR_CXX20 bool test() {
  check_cvs<int>();
  check_cvs<long>();
  check_cvs<double>();
  check_cvs<std::string>();
  check_cvs<A>();
  return true;
}

int main(int, char**) {
  static_assert(test());
  test();

  return 0;
}
