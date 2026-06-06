//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//
// This test pins down the ABI of std::pair with respect to being "trivially copyable".
//

// This test doesn't work when the deprecated ABI to turn off pair triviality is enabled.
// See libcxx/test/libcxx/utilities/utility/pairs/pairs.pair/abi.non_trivial_copy_move.pass.cpp instead.
// UNSUPPORTED: libcpp-deprecated-abi-disable-pair-trivial-copy-ctor

#include <type_traits>
#include <utility>

#include "test_macros.h"

struct trivially_copyable {
  int arr[4];
};

struct trivially_copyable_no_copy_assignment {
  int arr[4];
  trivially_copyable_no_copy_assignment& operator=(const trivially_copyable_no_copy_assignment&) = delete;
};
static_assert(std::is_trivially_copyable<trivially_copyable_no_copy_assignment>::value, "");

struct trivially_copyable_no_move_assignment {
  int arr[4];
  trivially_copyable_no_move_assignment& operator=(const trivially_copyable_no_move_assignment&) = delete;
};
static_assert(std::is_trivially_copyable<trivially_copyable_no_move_assignment>::value, "");

struct trivially_copyable_no_construction {
  int arr[4];
  trivially_copyable_no_construction()                                                     = default;
  trivially_copyable_no_construction(const trivially_copyable_no_construction&)            = delete;
  trivially_copyable_no_construction& operator=(const trivially_copyable_no_construction&) = default;
};
static_assert(std::is_trivially_copyable<trivially_copyable_no_construction>::value, "");

static_assert(!std::is_trivially_copyable<std::pair<int&, int> >::value, "");
static_assert(!std::is_trivially_copyable<std::pair<int, int&> >::value, "");
static_assert(!std::is_trivially_copyable<std::pair<int&, int&> >::value, "");

static_assert(!std::is_trivially_copyable<std::pair<int, int> >::value, "");
static_assert(!std::is_trivially_copyable<std::pair<int, char> >::value, "");
static_assert(!std::is_trivially_copyable<std::pair<char, int> >::value, "");
static_assert(!std::is_trivially_copyable<std::pair<std::pair<char, char>, int> >::value, "");
static_assert(!std::is_trivially_copyable<std::pair<trivially_copyable, int> >::value, "");
#if TEST_STD_VER == 03 // Known ABI difference
static_assert(!std::is_trivially_copyable<std::pair<trivially_copyable_no_copy_assignment, int> >::value, "");
static_assert(!std::is_trivially_copyable<std::pair<trivially_copyable_no_move_assignment, int> >::value, "");
#else
static_assert(std::is_trivially_copyable<std::pair<trivially_copyable_no_copy_assignment, int> >::value, "");
static_assert(std::is_trivially_copyable<std::pair<trivially_copyable_no_move_assignment, int> >::value, "");
#endif
static_assert(!std::is_trivially_copyable<std::pair<trivially_copyable_no_construction, int> >::value, "");

static_assert(std::is_trivially_copy_constructible<std::pair<int, int> >::value, "");
static_assert(std::is_trivially_move_constructible<std::pair<int, int> >::value, "");
static_assert(!std::is_trivially_copy_assignable<std::pair<int, int> >::value, "");
static_assert(!std::is_trivially_move_assignable<std::pair<int, int> >::value, "");
static_assert(std::is_trivially_destructible<std::pair<int, int> >::value, "");
