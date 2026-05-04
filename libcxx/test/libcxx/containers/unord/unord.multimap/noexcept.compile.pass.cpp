//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <unordered_map>

#include <type_traits>
#include <unordered_map>

#include "test_allocator.h"

template <class... Args>
inline constexpr bool is_constructible_but_throws =
    std::is_constructible<Args...>::value && !std::is_nothrow_constructible<Args...>::value;

namespace std_allocator {
using map        = std::unordered_multimap<int, int>;
using value_type = typename map::value_type;

static_assert(std::is_nothrow_constructible<map>::value);
static_assert(is_constructible_but_throws<map,
                                          int,
                                          const std::hash<int>&,
                                          const std::equal_to<int>&,
                                          const std::allocator<value_type>&>);
static_assert(is_constructible_but_throws<map,
                                          value_type*,
                                          value_type*,
                                          int,
                                          const std::hash<int>&,
                                          const std::equal_to<int>&,
                                          const std::allocator<value_type>&>);
#if TEST_STD_VER >= 23
static_assert(is_constructible_but_throws<map,
                                          std::from_range_t,
                                          map,
                                          int,
                                          const std::hash<int>&,
                                          const std::equal_to<int>&,
                                          const std::allocator<value_type>&>);
#endif
static_assert(is_constructible_but_throws<map, const std::allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<map, const map&, const std::allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<map, int, const std::allocator<std::pair<const int, int>>&>);
static_assert(
    is_constructible_but_throws<map, int, const std::hash<int>&, const std::allocator<std::pair<const int, int>>&>);
static_assert(
    is_constructible_but_throws<map, value_type*, value_type*, int, const std::allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<map,
                                          value_type*,
                                          value_type*,
                                          int,
                                          const std::hash<int>&,
                                          const std::allocator<std::pair<const int, int>>&>);
#if TEST_STD_VER >= 23
static_assert(is_constructible_but_throws<map,
                                          std::from_range_t,
                                          const map&,
                                          int,
                                          const std::allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<map,
                                          std::from_range_t,
                                          const map&,
                                          int,
                                          const std::hash<int>&,
                                          const std::allocator<std::pair<const int, int>>&>);
#endif
static_assert(is_constructible_but_throws<map,
                                          std::initializer_list<value_type>,
                                          int,
                                          const std::allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<map,
                                          std::initializer_list<value_type>,
                                          int,
                                          const std::hash<int>&,
                                          const std::allocator<std::pair<const int, int>>&>);
static_assert(std::is_nothrow_constructible<map, map&&>::value);

static_assert(std::is_nothrow_assignable<map&, map&&>::value);
static_assert(std::is_nothrow_swappable<map>::value);
} // namespace std_allocator

namespace test_alloc {
using map =
    std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, test_allocator<std::pair<const int, int>>>;
using value_type = typename map::value_type;

static_assert(std::is_nothrow_constructible<map>::value);
static_assert(is_constructible_but_throws<map,
                                          int,
                                          const std::hash<int>&,
                                          const std::equal_to<int>&,
                                          const test_allocator<value_type>&>);
static_assert(is_constructible_but_throws<map,
                                          value_type*,
                                          value_type*,
                                          int,
                                          const std::hash<int>&,
                                          const std::equal_to<int>&,
                                          const test_allocator<value_type>&>);
#if TEST_STD_VER >= 23
static_assert(is_constructible_but_throws<map,
                                          std::from_range_t,
                                          map,
                                          int,
                                          const std::hash<int>&,
                                          const std::equal_to<int>&,
                                          const test_allocator<value_type>&>);
#endif
static_assert(is_constructible_but_throws<map, const test_allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<map, const map&, const test_allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<map, int, const test_allocator<std::pair<const int, int>>&>);
static_assert(
    is_constructible_but_throws<map, int, const std::hash<int>&, const test_allocator<std::pair<const int, int>>&>);
static_assert(
    is_constructible_but_throws<map, value_type*, value_type*, int, const test_allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<map,
                                          value_type*,
                                          value_type*,
                                          int,
                                          const std::hash<int>&,
                                          const test_allocator<std::pair<const int, int>>&>);
#if TEST_STD_VER >= 23
static_assert(is_constructible_but_throws<map,
                                          std::from_range_t,
                                          const map&,
                                          int,
                                          const test_allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<map,
                                          std::from_range_t,
                                          const map&,
                                          int,
                                          const std::hash<int>&,
                                          const test_allocator<std::pair<const int, int>>&>);
#endif
static_assert(is_constructible_but_throws<map,
                                          std::initializer_list<value_type>,
                                          int,
                                          const test_allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<map,
                                          std::initializer_list<value_type>,
                                          int,
                                          const std::hash<int>&,
                                          const test_allocator<std::pair<const int, int>>&>);
static_assert(std::is_nothrow_constructible<map, map&&>::value);

static_assert(!std::is_nothrow_assignable<map&, map&&>::value);
static_assert(std::is_nothrow_swappable<map>::value);
} // namespace test_alloc
