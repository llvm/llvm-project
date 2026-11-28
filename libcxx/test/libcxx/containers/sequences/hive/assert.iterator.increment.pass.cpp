//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive
// REQUIRES: can-test-hardening-assertions-fast

// <hive>

// Incrementing an iterator that is end(), value-initialized, or on a slot with no element.

#include <hive>
#include <memory>

#include "check_assertion.h"

int main(int, char**) {
  using H = std::hive<int>;

  H empty;
  TEST_LIBCPP_ASSERT_FAILURE(++empty.end(), "hive: incrementing a non-dereferenceable iterator");
  TEST_LIBCPP_ASSERT_FAILURE(++empty.cend(), "hive: incrementing a non-dereferenceable iterator");
  TEST_LIBCPP_ASSERT_FAILURE(H::iterator()++, "hive: incrementing a non-dereferenceable iterator");

  H h(std::hive_limits(3, 3));
  h.insert(1);
  TEST_LIBCPP_ASSERT_FAILURE(++h.end(), "hive: incrementing a non-dereferenceable iterator");
  TEST_LIBCPP_ASSERT_FAILURE(h.end()++, "hive: incrementing a non-dereferenceable iterator");

  // A slot with no element: until erase leaves one behind, only the storage can name it.
  using Storage = std::__hive_storage<int, std::allocator<int>>;
  Storage s(std::hive_limits(3, 3), std::allocator<int>());
  auto [block, index] = s.__emplace(1);
  TEST_LIBCPP_ASSERT_FAILURE(
      ++std::__make_hive_iterator<Storage::iterator>(block, 2), "hive: incrementing a non-dereferenceable iterator");

  return 0;
}
