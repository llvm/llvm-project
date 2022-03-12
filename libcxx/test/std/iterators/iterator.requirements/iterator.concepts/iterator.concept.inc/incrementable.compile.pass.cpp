//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class In>
// concept indirectly_readable;

#include <iterator>

#include <concepts>
#include <memory>
#include <optional>

#include "../incrementable.h"

static_assert(std::incrementable<int>);
static_assert(std::incrementable<int*>);
static_assert(std::incrementable<int**>);

static_assert(!std::incrementable<postfix_increment_returns_void>);
static_assert(!std::incrementable<postfix_increment_returns_copy>);
static_assert(!std::incrementable<has_integral_minus>);
static_assert(!std::incrementable<has_distinct_difference_type_and_minus>);
static_assert(!std::incrementable<missing_difference_type>);
static_assert(!std::incrementable<floating_difference_type>);
static_assert(!std::incrementable<non_const_minus>);
static_assert(!std::incrementable<non_integral_minus>);
static_assert(!std::incrementable<bad_difference_type_good_minus>);
static_assert(!std::incrementable<not_default_initializable>);
static_assert(!std::incrementable<not_movable>);
static_assert(!std::incrementable<preinc_not_declared>);
static_assert(!std::incrementable<postinc_not_declared>);
static_assert(std::incrementable<incrementable_with_difference_type>);
static_assert(std::incrementable<incrementable_without_difference_type>);
static_assert(std::incrementable<difference_type_and_void_minus>);
static_assert(!std::incrementable<noncopyable_with_difference_type>);
static_assert(!std::incrementable<noncopyable_without_difference_type>);
static_assert(!std::incrementable<noncopyable_with_difference_type_and_minus>);
