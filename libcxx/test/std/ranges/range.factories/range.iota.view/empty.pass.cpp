//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// constexpr bool empty() const;

#include <cassert>
#include <concepts>
#include <limits>
#include <ranges>

#include "test_macros.h"
#include "types.h"

struct my_exception {};

struct ThrowBound {
  int bound;
  constexpr bool operator==(int x) const {
    if (x > 0) {
#if !defined(TEST_HAS_NO_EXCEPTIONS)
      throw my_exception{};
#else
      assert(false && "Exceptions are disabled");
#endif
    }
    return x == bound;
  }

  constexpr bool operator==(SomeInt x) const {
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    throw my_exception();
#else
    assert(false && "Exceptions are disabled");
#endif
    return x.value_ == bound;
  }
};

// clang-format off
constexpr bool test() {
  // Both are integer like and neither less than zero.
  {
    {
      const std::ranges::iota_view<int, int> io(1, 10);
      assert(!io.empty());
      LIBCPP_ASSERT_NOEXCEPT(io.empty());
    }
    {
      const std::ranges::iota_view<int, int> io(2, 2);
      assert(io.empty());
      LIBCPP_ASSERT_NOEXCEPT(io.empty());
    }
  }

  // Both are integer like and both are less than zero.
  {
    {
      const std::ranges::iota_view<int, int> io(-10, -5);
      assert(!io.empty());
      LIBCPP_ASSERT_NOEXCEPT(io.empty());
    }
    {
      const std::ranges::iota_view<int, int> io(-10, -10);
      assert(io.empty());
      LIBCPP_ASSERT_NOEXCEPT(io.empty());
    }
    {
      const std::ranges::iota_view<int, int> io(0, 0);
      assert(io.empty());
      LIBCPP_ASSERT_NOEXCEPT(io.empty());
    }
  }

  // Both are integer like and "value_" is less than zero.
  {
    const std::ranges::iota_view<int, int> io(-10, 10);
    assert(!io.empty());
    LIBCPP_ASSERT_NOEXCEPT(io.empty());
  }

  // Neither are integer like.
  {
    const std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(-20), SomeInt(-10));
    assert(!io.empty());
    LIBCPP_ASSERT_NOEXCEPT(io.empty());
  }
  {
    const std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(-10), SomeInt(-10));
    assert(io.empty());
    LIBCPP_ASSERT_NOEXCEPT(io.empty());
  }
  {
    const std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(0), SomeInt(0));
    assert(io.empty());
    LIBCPP_ASSERT_NOEXCEPT(io.empty());
  }
  {
    const std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(10), SomeInt(20));
    assert(!io.empty());
    LIBCPP_ASSERT_NOEXCEPT(io.empty());
  }
  {
    const std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(10), SomeInt(10));
    assert(io.empty());
    LIBCPP_ASSERT_NOEXCEPT(io.empty());
  }

  return true;
}
// clang-format on

void test_throw() {
#if !defined(TEST_HAS_NO_EXCEPTIONS)
  try {
    // Both are integer like and !noexcept(value_ == bound_)
    {
      const std::ranges::iota_view<int, ThrowBound> io(10, ThrowBound{10});
      assert(io.empty());
      LIBCPP_ASSERT_NOT_NOEXCEPT(io.empty());
    }

    // Neither are integer like and !noexcept(value_ == bound_)
    {
      const std::ranges::iota_view<SomeInt, ThrowBound> io(SomeInt{10}, ThrowBound{10});
      assert(io.empty());
      LIBCPP_ASSERT_NOT_NOEXCEPT(io.empty());
    }
  } catch (...) {
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main() {
  test();
  test_throw();
  static_assert(test());
  return 0;
}
