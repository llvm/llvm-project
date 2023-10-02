//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test the mandates
// constexpr T& value() & ;
// constexpr const T& value() const &;
// Mandates: is_copy_constructible_v<E> is true.

// constexpr T&& value() &&;
// constexpr const T&& value() const &&;
// Mandates: is_copy_constructible_v<E> is true and is_constructible_v<E, decltype(std::move(error()))> is true.

#include <expected>
#include <utility>

#include "MoveOnly.h"

struct CopyConstructible {
  constexpr CopyConstructible()                         = default;
  constexpr CopyConstructible(const CopyConstructible&) = default;
};

struct CopyConstructibleButNotMoveConstructible {
  constexpr CopyConstructibleButNotMoveConstructible()                                                 = default;
  constexpr CopyConstructibleButNotMoveConstructible(const CopyConstructibleButNotMoveConstructible&)  = default;
  constexpr CopyConstructibleButNotMoveConstructible(CopyConstructibleButNotMoveConstructible&&)       = delete;
  constexpr CopyConstructibleButNotMoveConstructible(const CopyConstructibleButNotMoveConstructible&&) = delete;
};

struct CopyConstructibleAndMoveConstructible {
  constexpr CopyConstructibleAndMoveConstructible()                                             = default;
  constexpr CopyConstructibleAndMoveConstructible(const CopyConstructibleAndMoveConstructible&) = default;
  constexpr CopyConstructibleAndMoveConstructible(CopyConstructibleAndMoveConstructible&&)      = default;
};

// clang-format off
void test() {

  // Test & overload
  {
    // is_copy_constructible_v<E> is true.
    {
      std::expected<int, CopyConstructible> e;
      [[maybe_unused]] auto val = e.value();
    }

    // is_copy_constructible_v<E> is false.
    {
      std::expected<int, MoveOnly> e;
      [[maybe_unused]] auto val = e.value();
      // expected-error-re@*:* {{static assertion failed {{.*}}error_type has to be copy constructible}}
    }
  }

  // Test const& overload
  {
    // is_copy_constructible_v<E> is true.
    {
      const std::expected<int, CopyConstructible> e;
      [[maybe_unused]] auto val = e.value();
    }

    // is_copy_constructible_v<E> is false.
    {
      const std::expected<int, MoveOnly> e;
      [[maybe_unused]] auto val = e.value();
      // expected-error-re@*:* {{static assertion failed {{.*}}error_type has to be copy constructible}}
    }
  }

  // Test && overload
  {
    // is_copy_constructible_v<E> is false.
    {
      std::expected<int, MoveOnly> e;
      [[maybe_unused]] auto val = std::move(e).value();
      // expected-error-re@*:* {{static assertion failed {{.*}}error_type has to be both copy constructible and constructible from decltype(std::move(error()))}}
    }

    // is_copy_constructible_v<E> is true and is_constructible_v<E, decltype(std::move(error()))> is true.
    {
      std::expected<int, CopyConstructibleAndMoveConstructible> e;
      [[maybe_unused]] auto val = std::move(e).value();
    }

    // is_copy_constructible_v<E> is true and is_constructible_v<E, decltype(std::move(error()))> is false.
    {
      std::expected<int, CopyConstructibleButNotMoveConstructible> e;
      [[maybe_unused]] auto val = std::move(e).value();
      // expected-error-re@*:* {{static assertion failed {{.*}}error_type has to be both copy constructible and constructible from decltype(std::move(error()))}}
    }
  }

  // Test const&& overload
  {
    // is_copy_constructible_v<E> is false.
    {
      const std::expected<int, MoveOnly> e;
      [[maybe_unused]] auto val = std::move(e).value();
      // expected-error-re@*:* {{static assertion failed {{.*}}error_type has to be both copy constructible and constructible from decltype(std::move(error()))}}
    }

    // is_copy_constructible_v<E> is true and is_constructible_v<E, decltype(std::move(error()))> is true.
    {
      const std::expected<int, CopyConstructibleAndMoveConstructible> e;
      [[maybe_unused]] auto val = std::move(e).value();
    }

    // is_copy_constructible_v<E> is true and is_constructible_v<E, decltype(std::move(error()))> is false.
    {
      const std::expected<int, CopyConstructibleButNotMoveConstructible> e;
      [[maybe_unused]] auto val = std::move(e).value();
      // expected-error-re@*:* {{static assertion failed {{.*}}error_type has to be both copy constructible and constructible from decltype(std::move(error()))}}
    }
  }
// These diagnostics happen when we try to construct bad_expected_access from the non copy-constructible error type.
#ifndef _LIBCPP_HAS_NO_EXCEPTIONS
  // expected-error-re@*:* {{call to deleted constructor of{{.*}}}}
  // expected-error-re@*:* {{call to deleted constructor of{{.*}}}}
  // expected-error-re@*:* {{call to deleted constructor of{{.*}}}}
  // expected-error-re@*:* {{call to deleted constructor of{{.*}}}}
#endif
}
// clang-format on
