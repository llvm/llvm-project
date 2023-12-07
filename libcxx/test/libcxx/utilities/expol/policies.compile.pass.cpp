//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure we don't allow copying the execution policies in any way to avoid users relying on it.

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

#include <execution>
#include <type_traits>

#include "test_macros.h"

static_assert(!std::is_default_constructible_v<std::execution::sequenced_policy>);
static_assert(!std::is_copy_constructible_v<std::execution::sequenced_policy>);
static_assert(!std::is_move_constructible_v<std::execution::sequenced_policy>);
static_assert(!std::is_copy_assignable_v<std::execution::sequenced_policy>);
static_assert(!std::is_move_assignable_v<std::execution::sequenced_policy>);

static_assert(!std::is_default_constructible_v<std::execution::parallel_policy>);
static_assert(!std::is_copy_constructible_v<std::execution::parallel_policy>);
static_assert(!std::is_move_constructible_v<std::execution::parallel_policy>);
static_assert(!std::is_copy_assignable_v<std::execution::parallel_policy>);
static_assert(!std::is_move_assignable_v<std::execution::parallel_policy>);

static_assert(!std::is_default_constructible_v<std::execution::parallel_unsequenced_policy>);
static_assert(!std::is_copy_constructible_v<std::execution::parallel_unsequenced_policy>);
static_assert(!std::is_move_constructible_v<std::execution::parallel_unsequenced_policy>);
static_assert(!std::is_copy_assignable_v<std::execution::parallel_unsequenced_policy>);
static_assert(!std::is_move_assignable_v<std::execution::parallel_unsequenced_policy>);

#if TEST_STD_VER >= 20
static_assert(!std::is_default_constructible_v<std::execution::unsequenced_policy>);
static_assert(!std::is_copy_constructible_v<std::execution::unsequenced_policy>);
static_assert(!std::is_move_constructible_v<std::execution::unsequenced_policy>);
static_assert(!std::is_copy_assignable_v<std::execution::unsequenced_policy>);
static_assert(!std::is_move_assignable_v<std::execution::unsequenced_policy>);
#endif
