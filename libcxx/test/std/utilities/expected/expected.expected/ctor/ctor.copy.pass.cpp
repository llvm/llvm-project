//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr expected(const expected& rhs);
//
// Effects: If rhs.has_value() is true, direct-non-list-initializes val with *rhs.
// Otherwise, direct-non-list-initializes unex with rhs.error().
//
// Postconditions: rhs.has_value() == this->has_value().
//
// Throws: Any exception thrown by the initialization of val or unex.
//
// Remarks: This constructor is defined as deleted unless
// - is_copy_constructible_v<T> is true and
// - is_copy_constructible_v<E> is true.
//
// This constructor is trivial if
// - is_trivially_copy_constructible_v<T> is true and
// - is_trivially_copy_constructible_v<E> is true.

#include <cassert>
#include <expected>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "../../types.h"

struct NonCopyable {
  NonCopyable(const NonCopyable&) = delete;
};

struct CopyableNonTrivial {
  int i;
  constexpr CopyableNonTrivial(int ii) : i(ii) {}
  constexpr CopyableNonTrivial(const CopyableNonTrivial& o) { i = o.i; }
  friend constexpr bool operator==(const CopyableNonTrivial&, const CopyableNonTrivial&) = default;
};

// Test: This constructor is defined as deleted unless
// - is_copy_constructible_v<T> is true and
// - is_copy_constructible_v<E> is true.
static_assert(std::is_copy_constructible_v<std::expected<int, int>>);
static_assert(std::is_copy_constructible_v<std::expected<CopyableNonTrivial, int>>);
static_assert(std::is_copy_constructible_v<std::expected<int, CopyableNonTrivial>>);
static_assert(std::is_copy_constructible_v<std::expected<CopyableNonTrivial, CopyableNonTrivial>>);
static_assert(!std::is_copy_constructible_v<std::expected<NonCopyable, int>>);
static_assert(!std::is_copy_constructible_v<std::expected<int, NonCopyable>>);
static_assert(!std::is_copy_constructible_v<std::expected<NonCopyable, NonCopyable>>);

// Test: This constructor is trivial if
// - is_trivially_copy_constructible_v<T> is true and
// - is_trivially_copy_constructible_v<E> is true.
static_assert(std::is_trivially_copy_constructible_v<std::expected<int, int>>);
static_assert(!std::is_trivially_copy_constructible_v<std::expected<CopyableNonTrivial, int>>);
static_assert(!std::is_trivially_copy_constructible_v<std::expected<int, CopyableNonTrivial>>);
static_assert(!std::is_trivially_copy_constructible_v<std::expected<CopyableNonTrivial, CopyableNonTrivial>>);

struct Any {
  constexpr Any()                      = default;
  constexpr Any(const Any&)            = default;
  constexpr Any& operator=(const Any&) = default;

  template <class T>
    requires(!std::is_same_v<Any, std::decay_t<T>> && std::is_copy_constructible_v<std::decay_t<T>>)
  constexpr Any(T&&) {}
};

constexpr bool test() {
  // copy the value non-trivial
  {
    const std::expected<CopyableNonTrivial, int> e1(5);
    auto e2 = e1;
    assert(e2.has_value());
    assert(e2.value().i == 5);
  }

  // copy the error non-trivial
  {
    const std::expected<int, CopyableNonTrivial> e1(std::unexpect, 5);
    auto e2 = e1;
    assert(!e2.has_value());
    assert(e2.error().i == 5);
  }

  // copy the value trivial
  {
    const std::expected<int, int> e1(5);
    auto e2 = e1;
    assert(e2.has_value());
    assert(e2.value() == 5);
  }

  // copy the error trivial
  {
    const std::expected<int, int> e1(std::unexpect, 5);
    auto e2 = e1;
    assert(!e2.has_value());
    assert(e2.error() == 5);
  }

  // copy TailClobberer as value
  {
    const std::expected<TailClobberer<0>, bool> e1;
    auto e2 = e1;
    assert(e2.has_value());
  }

  // copy TailClobberer as error
  {
    const std::expected<bool, TailClobberer<1>> e1(std::unexpect);
    auto e2 = e1;
    assert(!e2.has_value());
  }

  {
    // TODO(LLVM 20): Remove once we drop support for Clang 17
#if defined(TEST_CLANG_VER) && TEST_CLANG_VER >= 1800
    // https://github.com/llvm/llvm-project/issues/92676
    std::expected<Any, int> e1;
    auto e2 = e1;
    assert(e2.has_value());
#endif
  }

  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Throwing {
    Throwing() = default;
    Throwing(const Throwing&) { throw Except{}; }
  };

  // throw on copying value
  {
    const std::expected<Throwing, int> e1;
    try {
      [[maybe_unused]] auto e2 = e1;
      assert(false);
    } catch (Except) {
    }
  }

  // throw on copying error
  {
    const std::expected<int, Throwing> e1(std::unexpect);
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
