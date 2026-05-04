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
#include <map>

#include "test_allocator.h"

template <class... Args>
inline constexpr bool is_constructible_but_throws =
    std::is_constructible<Args...>::value && !std::is_nothrow_constructible<Args...>::value;

namespace std_allocator {
using map        = std::map<int, int>;
using value_type = typename map::value_type;

static_assert(std::is_nothrow_constructible<map>::value);
static_assert(is_constructible_but_throws<map, const std::less<int>&, const std::allocator<value_type>&>);
static_assert(is_constructible_but_throws<map,
                                          value_type*,
                                          value_type*,
                                          const std::less<int>&,
                                          const std::allocator<value_type>&>);
#if TEST_STD_VER >= 23
static_assert(
    is_constructible_but_throws<map, std::from_range_t, map, const std::less<int>&, const std::allocator<value_type>&>);
#endif
static_assert(is_constructible_but_throws<map, const std::allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<map, const map&, const std::allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<map, map&&, const std::allocator<std::pair<const int, int>>&>);
static_assert(std::is_nothrow_constructible<map, map&&>::value);

static_assert(std::is_nothrow_assignable<map&, map&&>::value);
static_assert(std::is_nothrow_swappable<map>::value);

using multimap = std::multimap<int, int>;

static_assert(std::is_nothrow_constructible<multimap>::value);
static_assert(is_constructible_but_throws<multimap, const std::less<int>&, const std::allocator<value_type>&>);
static_assert(is_constructible_but_throws<multimap,
                                          value_type*,
                                          value_type*,
                                          const std::less<int>&,
                                          const std::allocator<value_type>&>);
#if TEST_STD_VER >= 23
static_assert(is_constructible_but_throws<multimap,
                                          std::from_range_t,
                                          multimap,
                                          const std::less<int>&,
                                          const std::allocator<value_type>&>);
#endif
static_assert(is_constructible_but_throws<multimap, const std::allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<multimap, const multimap&, const std::allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<multimap, multimap&&, const std::allocator<std::pair<const int, int>>&>);
static_assert(std::is_nothrow_constructible<multimap, multimap&&>::value);

static_assert(std::is_nothrow_assignable<multimap&, multimap&&>::value);
static_assert(std::is_nothrow_swappable<multimap>::value);
} // namespace std_allocator

namespace test_alloc {
using map        = std::map<int, int, std::less<int>, test_allocator<std::pair<const int, int>>>;
using value_type = typename map::value_type;

static_assert(std::is_nothrow_constructible<map>::value);
static_assert(is_constructible_but_throws<map, const std::less<int>&, const test_allocator<value_type>&>);
static_assert(is_constructible_but_throws<map,
                                          value_type*,
                                          value_type*,
                                          const std::less<int>&,
                                          const test_allocator<value_type>&>);
#if TEST_STD_VER >= 23
static_assert(
    is_constructible_but_throws<map, std::from_range_t, map, const std::less<int>&, const test_allocator<value_type>&>);
#endif
static_assert(is_constructible_but_throws<map, const test_allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<map, const map&, const test_allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<map, map&&, const test_allocator<std::pair<const int, int>>&>);
static_assert(std::is_nothrow_constructible<map, map&&>::value);

static_assert(!std::is_nothrow_assignable<map&, map&&>::value);
static_assert(std::is_nothrow_swappable<map>::value);

using multimap = std::multimap<int, int, std::less<int>, test_allocator<std::pair<const int, int>>>;

static_assert(std::is_nothrow_constructible<multimap>::value);
static_assert(is_constructible_but_throws<multimap, const std::less<int>&, const test_allocator<value_type>&>);
static_assert(is_constructible_but_throws<multimap,
                                          value_type*,
                                          value_type*,
                                          const std::less<int>&,
                                          const test_allocator<value_type>&>);
#if TEST_STD_VER >= 23
static_assert(is_constructible_but_throws<multimap,
                                          std::from_range_t,
                                          multimap,
                                          const std::less<int>&,
                                          const test_allocator<value_type>&>);
#endif
static_assert(is_constructible_but_throws<multimap, const test_allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<multimap, const multimap&, const test_allocator<std::pair<const int, int>>&>);
static_assert(is_constructible_but_throws<multimap, multimap&&, const test_allocator<std::pair<const int, int>>&>);
static_assert(std::is_nothrow_constructible<multimap, multimap&&>::value);

static_assert(!std::is_nothrow_assignable<multimap&, multimap&&>::value);
static_assert(std::is_nothrow_swappable<multimap>::value);
} // namespace test_alloc
