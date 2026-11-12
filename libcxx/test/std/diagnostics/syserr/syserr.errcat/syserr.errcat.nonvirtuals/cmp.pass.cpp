//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <system_error>

// class error_category

// strong_ordering operator<=>(const error_category& rhs) const noexcept;

#include <system_error>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**) {
  AssertOrderAreNoexcept<std::error_category>();
  AssertOrderReturn<std::strong_ordering, std::error_category>();

  const std::error_category& e_cat1 = std::generic_category();
  const std::error_category& e_cat2 = std::generic_category();
  const std::error_category& e_cat3 = std::system_category();

  assert(testOrder(e_cat1, e_cat2, std::strong_ordering::equal));

  bool isLess = e_cat1 < e_cat3;
  assert(testOrder(e_cat1, e_cat3, isLess ? std::strong_ordering::less : std::strong_ordering::greater));

  return 0;
}
