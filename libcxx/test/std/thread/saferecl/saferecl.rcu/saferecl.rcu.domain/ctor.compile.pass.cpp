//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// REQUIRES: std-at-least-c++26

// rcu_domain(const rcu_domain&) = delete;
// rcu_domain& operator=(const rcu_domain&) = delete;

#include <rcu>
#include <type_traits>

static_assert(!std::is_default_constructible_v<std::rcu_domain>);
static_assert(!std::is_copy_constructible_v<std::rcu_domain>);
static_assert(!std::is_move_constructible_v<std::rcu_domain>);
static_assert(!std::is_copy_assignable_v<std::rcu_domain>);
static_assert(!std::is_move_assignable_v<std::rcu_domain>);
