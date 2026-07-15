//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <flat_map>

// flat_multimap(key_container_type , mapped_container_type , const key_compare& __comp = key_compare())
// flat_multimap(const key_container_type& , const mapped_container_type& , const _Allocator& )
// flat_multimap(const key_container_type& , const mapped_container_type& , const key_compare&, const _Allocator& )
// void replace(key_container_type&& , mapped_container_type&&)
//

#include <flat_map>
#include <functional>
#include <memory>
#include <vector>

#include "check_assertion.h"

int main(int, char**) {
  using M = std::flat_multimap<int, int>;

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] { M m({1, 2, 3}, {4}); }()), "flat_multimap keys and mapped containers have different size");

  TEST_LIBCPP_ASSERT_FAILURE(([] { M m({1, 2, 3}, {4}, std::less<int>{}); }()),
                             "flat_multimap keys and mapped containers have different size");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector keys{1, 2, 3};
        const std::vector values{4};
        const std::allocator<int> alloc{};
        M m(keys, values, alloc);
      }()),
      "flat_multimap keys and mapped containers have different size");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector keys{1, 2, 3};
        const std::vector values{4};
        const std::less<int> key_compare{};
        const std::allocator<int> alloc{};
        M m(keys, values, key_compare, alloc);
      }()),
      "flat_multimap keys and mapped containers have different size");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::vector keys{1, 2, 3};
        std::vector values{4};
        M m;
        m.replace(std::move(keys), std::move(values));
      }()),
      "flat_multimap keys and mapped containers have different size");

  return 0;
}
