//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr expected(const expected& rhs);
//
// Effects: If rhs.has_value() is false, direct-non-list-initializes unex with rhs.error().
//
// Postconditions: rhs.has_value() == this->has_value().
//
// Throws: Any exception thrown by the initialization of unex.
//
// Remarks:
// - This constructor is defined as deleted unless is_copy_constructible_v<E> is true.
// - This constructor is trivial if is_trivially_copy_constructible_v<E> is true.

#include <cassert>
#include <expected>
#include <type_traits>
#include <utility>

#include "test_macros.h"

struct NonCopyable {
  NonCopyable(const NonCopyable&) = delete;
};

struct CopyableNonTrivial {
  int i;
  constexpr CopyableNonTrivial(int ii) : i(ii) {}
  constexpr CopyableNonTrivial(const CopyableNonTrivial& o) { i = o.i; }
  friend constexpr bool operator==(const CopyableNonTrivial&, const CopyableNonTrivial&) = default;
};

// Test: This constructor is defined as deleted unless is_copy_constructible_v<E> is true.
static_assert(std::is_copy_constructible_v<std::expected<void, int>>);
static_assert(std::is_copy_constructible_v<std::expected<void, CopyableNonTrivial>>);
static_assert(!std::is_copy_constructible_v<std::expected<void, NonCopyable>>);

// Test: This constructor is trivial if is_trivially_copy_constructible_v<E> is true.
static_assert(std::is_trivially_copy_constructible_v<std::expected<void, int>>);
static_assert(!std::is_trivially_copy_constructible_v<std::expected<void, CopyableNonTrivial>>);

constexpr bool test() {
  // copy the error non-trivial
  {
    const std::expected<void, CopyableNonTrivial> e1(std::unexpect, 5);
    auto e2 = e1;
    assert(!e2.has_value());
    assert(e2.error().i == 5);
  }

  // copy the error trivial
  {
    const std::expected<void, int> e1(std::unexpect, 5);
    auto e2 = e1;
    assert(!e2.has_value());
    assert(e2.error() == 5);
  }
  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Except {};

  struct Throwing {
    Throwing() = default;
    Throwing(const Throwing&) { throw Except{}; }
  };

  // throw on copying error
  {
    const std::expected<void, Throwing> e1(std::unexpect);
    try {
      [[maybe_unused]] auto e2 = e1;
      assert(false);
    } catch (Except) {
    }
  }

#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test();
  static_assert(test());
  testException();
  return 0;
}
