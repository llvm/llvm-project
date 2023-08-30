//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: !libcpp-has-hardened-mode && !libcpp-has-debug-mode && !libcpp-has-assertions
// XFAIL: availability-verbose_abort-missing

// constexpr const T& operator*() const & noexcept;
// constexpr T& operator*() & noexcept;
// constexpr T&& operator*() && noexcept;
// constexpr const T&& operator*() const && noexcept;
//
// Preconditions: has_value() is true.

#include <expected>
#include <utility>

#include "check_assertion.h"

int main(int, char**) {
  std::expected<int, int> e{std::unexpect, 5};
  TEST_LIBCPP_ASSERT_FAILURE(*e, "expected::operator* requires the expected to contain a value");
  TEST_LIBCPP_ASSERT_FAILURE(*std::as_const(e), "expected::operator* requires the expected to contain a value");
  TEST_LIBCPP_ASSERT_FAILURE(*std::move(e), "expected::operator* requires the expected to contain a value");
  TEST_LIBCPP_ASSERT_FAILURE(*std::move(std::as_const(e)), "expected::operator* requires the expected to contain a value");

  return 0;
}
