//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// template<class T> struct is_execution_policy;
// template<class T> constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

// UNSUPPORTED: c++03, c++11, c++14

// REQUIRES: with-pstl

#include <execution>

#include "test_macros.h"

static_assert(std::is_execution_policy<std::execution::sequenced_policy>::value);
static_assert(std::is_execution_policy_v<std::execution::sequenced_policy>);
static_assert(std::is_execution_policy<std::execution::parallel_policy>::value);
static_assert(std::is_execution_policy_v<std::execution::parallel_policy>);
static_assert(std::is_execution_policy<std::execution::parallel_unsequenced_policy>::value);
static_assert(std::is_execution_policy_v<std::execution::parallel_unsequenced_policy>);

#if TEST_STD_VER >= 20
static_assert(std::is_execution_policy<std::execution::unsequenced_policy>::value);
static_assert(std::is_execution_policy_v<std::execution::unsequenced_policy>);
#endif
