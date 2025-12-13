//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-hardening-mode=none
// REQUIRES: libcpp-hardening-mode=debug
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <flat_set>

// flat_multiset(container_type , const key_compare& __comp = key_compare())
// flat_multiset(const container_type&  , const _Allocator& )
// flat_multiset(const container_type&  , const key_compare&, const _Allocator& )
// void replace(container_type&& )
//

#include <flat_set>
#include <functional>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "check_assertion.h"

int main(int, char**) {
  using M = std::flat_multiset<int>;

  TEST_LIBCPP_ASSERT_FAILURE(([] { M m(std::sorted_equivalent, {4, 2, 3}); }()), "Key container is not sorted");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] { M m(std::sorted_equivalent, {4, 2, 3}, std::less<int>{}); }()), "Key container is not sorted");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector keys{4, 2, 3};
        const std::allocator<int> alloc{};
        M m(std::sorted_equivalent, keys, alloc);
      }()),
      "Key container is not sorted");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector keys{4, 2, 3};
        const std::allocator<int> alloc{};
        const std::less<int> comp{};
        M m(std::sorted_equivalent, keys, comp, alloc);
      }()),
      "Key container is not sorted");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector<int> v{4, 2, 3};
        const std::less<int> comp{};
        M m(std::sorted_equivalent, v.begin(), v.end(), comp);
      }()),
      "Key container is not sorted");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector<int> v{4, 2, 3};
        const std::less<int> comp{};
        const std::allocator<int> alloc{};
        M m(std::sorted_equivalent, v.begin(), v.end(), comp, alloc);
      }()),
      "Key container is not sorted");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector<int> v{4, 2, 3};
        const std::allocator<int> alloc{};
        M m(std::sorted_equivalent, v.begin(), v.end(), alloc);
      }()),
      "Key container is not sorted");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::initializer_list<int> v{4, 2, 3};
        const std::less<int> comp{};
        M m(std::sorted_equivalent, v, comp);
      }()),
      "Key container is not sorted");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::initializer_list<int> v{4, 2, 3};
        const std::less<int> comp{};
        const std::allocator<int> alloc{};
        M m(std::sorted_equivalent, v, comp, alloc);
      }()),
      "Key container is not sorted");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::initializer_list<int> v{4, 2, 3};
        const std::allocator<int> alloc{};
        M m(std::sorted_equivalent, v, alloc);
      }()),
      "Key container is not sorted");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector<int> v{4, 2, 3};
        M m;
        m.insert(std::sorted_equivalent, v.begin(), v.end());
      }()),
      "Key container is not sorted");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::initializer_list<int> v{4, 2, 3};
        M m;
        m.insert(std::sorted_equivalent, v);
      }()),
      "Key container is not sorted");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::vector keys{2, 1, 3};
        M m;
        m.replace(std::move(keys));
      }()),
      "Key container is not sorted");

  return 0;
}
