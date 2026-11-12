//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <system_error>

// class error_code

// strong_ordering operator<=>(const error_code& lhs, const error_code& rhs) noexcept

#include <system_error>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**) {
  AssertOrderAreNoexcept<std::error_code>();
  AssertOrderReturn<std::strong_ordering, std::error_code>();

  // Same error category
  std::error_code ec1a = std::error_code(1, std::generic_category());
  std::error_code ec1b = std::error_code(1, std::generic_category());
  std::error_code ec2  = std::error_code(2, std::generic_category());

  assert(testOrder(ec1a, ec1b, std::strong_ordering::equal));
  assert(testOrder(ec1a, ec2, std::strong_ordering::less));

  // Different error category
  const std::error_code& ec3 = std::error_code(2, std::system_category());

  bool isLess = ec2 < ec3;
  assert(testOrder(ec2, ec3, isLess ? std::strong_ordering::less : std::strong_ordering::greater));

  return 0;
}
