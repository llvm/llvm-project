//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// XFAIL: availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

// constexpr const E& error() const & noexcept;
// constexpr E& error() & noexcept;
// constexpr E&& error() && noexcept;
// constexpr const E&& error() const && noexcept;
//
// Preconditions: has_value() is false.

#include <expected>
#include <utility>

#include "check_assertion.h"

int main(int, char**) {
  std::expected<void, int> e{std::in_place};
  TEST_LIBCPP_ASSERT_FAILURE(e.error(), "expected::error requires the expected to contain an error");
  TEST_LIBCPP_ASSERT_FAILURE(std::as_const(e).error(), "expected::error requires the expected to contain an error");
  TEST_LIBCPP_ASSERT_FAILURE(std::move(e).error(), "expected::error requires the expected to contain an error");
  TEST_LIBCPP_ASSERT_FAILURE(std::move(std::as_const(e)).error(), "expected::error requires the expected to contain an error");

  return 0;
}
