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

// <flat_map>

// flat_map(key_container_type , mapped_container_type , const key_compare& __comp = key_compare())
// flat_map(const key_container_type& , const mapped_container_type& , const _Allocator& )
// flat_map(const key_container_type& , const mapped_container_type& , const key_compare&, const _Allocator& )
// void replace(key_container_type&& , mapped_container_type&&)
//

#include <flat_map>
#include <functional>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "check_assertion.h"

int main(int, char**) {
  using M = std::flat_map<int, int>;

  TEST_LIBCPP_ASSERT_FAILURE(([] { M m(std::sorted_unique, {2, 2, 3}, {4, 5, 6}); }()),
                             "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(([] { M m(std::sorted_unique, {4, 2, 3}, {4, 5, 6}); }()),
                             "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(([] { M m(std::sorted_unique, {2, 2, 3}, {4, 5, 6}, std::less<int>{}); }()),
                             "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(([] { M m(std::sorted_unique, {4, 2, 3}, {4, 5, 6}, std::less<int>{}); }()),
                             "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector keys{2, 2, 3};
        const std::vector values{4, 5, 6};
        const std::allocator<int> alloc{};
        M m(std::sorted_unique, keys, values, alloc);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector keys{4, 2, 3};
        const std::vector values{4, 5, 6};
        const std::allocator<int> alloc{};
        M m(std::sorted_unique, keys, values, alloc);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector keys{2, 2, 3};
        const std::vector values{4, 5, 6};
        const std::allocator<int> alloc{};
        const std::less<int> comp{};
        M m(std::sorted_unique, keys, values, comp, alloc);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector keys{4, 2, 3};
        const std::vector values{4, 5, 6};
        const std::allocator<int> alloc{};
        const std::less<int> comp{};
        M m(std::sorted_unique, keys, values, comp, alloc);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector<std::pair<int, int>> v{{2, 4}, {2, 5}, {3, 6}};
        const std::less<int> comp{};
        M m(std::sorted_unique, v.begin(), v.end(), comp);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector<std::pair<int, int>> v{{4, 4}, {2, 5}, {3, 6}};
        const std::less<int> comp{};
        M m(std::sorted_unique, v.begin(), v.end(), comp);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector<std::pair<int, int>> v{{2, 4}, {2, 5}, {3, 6}};
        const std::less<int> comp{};
        const std::allocator<int> alloc{};
        M m(std::sorted_unique, v.begin(), v.end(), comp, alloc);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector<std::pair<int, int>> v{{4, 4}, {2, 5}, {3, 6}};
        const std::less<int> comp{};
        const std::allocator<int> alloc{};
        M m(std::sorted_unique, v.begin(), v.end(), comp, alloc);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector<std::pair<int, int>> v{{2, 4}, {2, 5}, {3, 6}};
        const std::allocator<int> alloc{};
        M m(std::sorted_unique, v.begin(), v.end(), alloc);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector<std::pair<int, int>> v{{4, 4}, {2, 5}, {3, 6}};
        const std::allocator<int> alloc{};
        M m(std::sorted_unique, v.begin(), v.end(), alloc);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::initializer_list<std::pair<int, int>> v{{2, 4}, {2, 5}, {3, 6}};
        const std::less<int> comp{};
        M m(std::sorted_unique, v, comp);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::initializer_list<std::pair<int, int>> v{{4, 4}, {2, 5}, {3, 6}};
        const std::less<int> comp{};
        M m(std::sorted_unique, v, comp);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::initializer_list<std::pair<int, int>> v{{2, 4}, {2, 5}, {3, 6}};
        const std::less<int> comp{};
        const std::allocator<int> alloc{};
        M m(std::sorted_unique, v, comp, alloc);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::initializer_list<std::pair<int, int>> v{{4, 4}, {2, 5}, {3, 6}};
        const std::less<int> comp{};
        const std::allocator<int> alloc{};
        M m(std::sorted_unique, v, comp, alloc);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::initializer_list<std::pair<int, int>> v{{2, 4}, {2, 5}, {3, 6}};
        const std::allocator<int> alloc{};
        M m(std::sorted_unique, v, alloc);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::initializer_list<std::pair<int, int>> v{{4, 4}, {2, 5}, {3, 6}};
        const std::allocator<int> alloc{};
        M m(std::sorted_unique, v, alloc);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector<std::pair<int, int>> v{{2, 4}, {2, 5}, {3, 6}};
        M m;
        m.insert(std::sorted_unique, v.begin(), v.end());
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        const std::vector<std::pair<int, int>> v{{4, 4}, {2, 5}, {3, 6}};
        M m;
        m.insert(std::sorted_unique, v.begin(), v.end());
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::initializer_list<std::pair<int, int>> v{{2, 4}, {2, 5}, {3, 6}};
        M m;
        m.insert(std::sorted_unique, v);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::initializer_list<std::pair<int, int>> v{{4, 4}, {2, 5}, {3, 6}};
        M m;
        m.insert(std::sorted_unique, v);
      }()),
      "Either the key container is not sorted or it contains duplicates");

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        std::vector keys{1, 1, 3};
        std::vector values{4, 5, 6};
        M m;
        m.replace(std::move(keys), std::move(values));
      }()),
      "Either the key container is not sorted or it contains duplicates");
  return 0;
}
