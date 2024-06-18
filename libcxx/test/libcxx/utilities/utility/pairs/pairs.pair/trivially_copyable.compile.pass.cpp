//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: gcc && (c++11 || c++14 || c++17)

#include <type_traits>
#include <utility>

#include "test_macros.h"

struct trivially_copyable {
  int arr[4];
};

struct trivially_copyable_no_assignment {
  int arr[4];
  trivially_copyable_no_assignment& operator=(const trivially_copyable_no_assignment&) = delete;
};
static_assert(std::is_trivially_copyable<trivially_copyable_no_assignment>::value, "");

static_assert(std::is_trivially_copy_constructible<std::pair<trivially_copyable_no_assignment, int> >::value, "");
static_assert(std::is_trivially_move_constructible<std::pair<trivially_copyable_no_assignment, int> >::value, "");
#if TEST_STD_VER >= 11 // This is https://llvm.org/PR90605
static_assert(!std::is_trivially_copy_assignable<std::pair<trivially_copyable_no_assignment, int> >::value, "");
static_assert(!std::is_trivially_move_assignable<std::pair<trivially_copyable_no_assignment, int> >::value, "");
#endif // TEST_STD_VER >= 11
static_assert(std::is_trivially_destructible<std::pair<trivially_copyable_no_assignment, int> >::value, "");

struct trivially_copyable_no_construction {
  int arr[4];
  trivially_copyable_no_construction()                                                     = default;
  trivially_copyable_no_construction(const trivially_copyable_no_construction&)            = delete;
  trivially_copyable_no_construction& operator=(const trivially_copyable_no_construction&) = default;
};
static_assert(std::is_trivially_copyable<trivially_copyable_no_construction>::value, "");

#ifdef _LIBCPP_ABI_TRIVIALLY_COPYABLE_PAIR
static_assert(!std::is_trivially_copy_constructible<std::pair<trivially_copyable_no_construction, int> >::value, "");
static_assert(!std::is_trivially_move_constructible<std::pair<trivially_copyable_no_construction, int> >::value, "");
static_assert(std::is_trivially_copy_assignable<std::pair<trivially_copyable_no_construction, int> >::value, "");
static_assert(std::is_trivially_move_assignable<std::pair<trivially_copyable_no_construction, int> >::value, "");
static_assert(std::is_trivially_destructible<std::pair<trivially_copyable_no_construction, int> >::value, "");
#else // _LIBCPP_ABI_TRIVIALLY_COPYABLE_PAIR
#  if TEST_STD_VER >= 11
static_assert(!std::is_trivially_copy_constructible<std::pair<trivially_copyable_no_construction, int> >::value, "");
static_assert(!std::is_trivially_move_constructible<std::pair<trivially_copyable_no_construction, int> >::value, "");
#  endif // TEST_STD_VER >= 11
static_assert(!std::is_trivially_copy_assignable<std::pair<trivially_copyable_no_construction, int> >::value, "");
static_assert(!std::is_trivially_move_assignable<std::pair<trivially_copyable_no_construction, int> >::value, "");
static_assert(std::is_trivially_destructible<std::pair<trivially_copyable_no_construction, int> >::value, "");
#endif   // _LIBCPP_ABI_TRIVIALLY_COPYABLE_PAIR

static_assert(!std::is_trivially_copyable<std::pair<int&, int> >::value, "");
static_assert(!std::is_trivially_copyable<std::pair<int, int&> >::value, "");
static_assert(!std::is_trivially_copyable<std::pair<int&, int&> >::value, "");

#ifdef _LIBCPP_ABI_TRIVIALLY_COPYABLE_PAIR
static_assert(std::is_trivially_copyable<std::pair<int, int> >::value, "");
static_assert(std::is_trivially_copyable<std::pair<int, char> >::value, "");
static_assert(std::is_trivially_copyable<std::pair<char, int> >::value, "");
static_assert(std::is_trivially_copyable<std::pair<std::pair<char, char>, int> >::value, "");
static_assert(std::is_trivially_copyable<std::pair<trivially_copyable, int> >::value, "");
static_assert(std::is_trivially_copyable<std::pair<trivially_copyable_no_assignment, int> >::value, "");
static_assert(std::is_trivially_copyable<std::pair<trivially_copyable_no_construction, int> >::value, "");

static_assert(std::is_trivially_copy_constructible<std::pair<int, int> >::value, "");
static_assert(std::is_trivially_move_constructible<std::pair<int, int> >::value, "");
static_assert(std::is_trivially_copy_assignable<std::pair<int, int> >::value, "");
static_assert(std::is_trivially_move_assignable<std::pair<int, int> >::value, "");
static_assert(std::is_trivially_destructible<std::pair<int, int> >::value, "");

#else // _LIBCPP_ABI_TRIVIALLY_COPYABLE_PAIR
static_assert(!std::is_trivially_copyable<std::pair<int, int> >::value, "");
static_assert(!std::is_trivially_copyable<std::pair<int, char> >::value, "");
static_assert(!std::is_trivially_copyable<std::pair<char, int> >::value, "");
static_assert(!std::is_trivially_copyable<std::pair<std::pair<char, char>, int> >::value, "");
static_assert(!std::is_trivially_copyable<std::pair<trivially_copyable, int> >::value, "");
#  if (TEST_STD_VER == 03 || TEST_STD_VER >= 23) && !defined(TEST_COMPILER_GCC)
static_assert(!std::is_trivially_copyable<std::pair<trivially_copyable_no_assignment, int> >::value, "");
#  else
static_assert(std::is_trivially_copyable<std::pair<trivially_copyable_no_assignment, int> >::value, "");
#  endif
static_assert(!std::is_trivially_copyable<std::pair<trivially_copyable_no_construction, int> >::value, "");

static_assert(std::is_trivially_copy_constructible<std::pair<int, int> >::value, "");
static_assert(std::is_trivially_move_constructible<std::pair<int, int> >::value, "");
static_assert(!std::is_trivially_copy_assignable<std::pair<int, int> >::value, "");
static_assert(!std::is_trivially_move_assignable<std::pair<int, int> >::value, "");
static_assert(std::is_trivially_destructible<std::pair<int, int> >::value, "");
#endif // _LIBCPP_ABI_TRIVIALLY_COPYABLE_PAIR
