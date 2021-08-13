//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class In>
// concept std::weakly_incrementable;

#include <iterator>

#include <concepts>
#include <memory>
#include <optional>

#include "../incrementable.h"

static_assert(std::weakly_incrementable<int>);
static_assert(std::weakly_incrementable<int*>);
static_assert(std::weakly_incrementable<int**>);
static_assert(!std::weakly_incrementable<int[]>);
static_assert(!std::weakly_incrementable<int[10]>);
static_assert(!std::weakly_incrementable<double>);
static_assert(!std::weakly_incrementable<int&>);
static_assert(!std::weakly_incrementable<int()>);
static_assert(!std::weakly_incrementable<int (*)()>);
static_assert(!std::weakly_incrementable<int (&)()>);

struct S {};
static_assert(!std::weakly_incrementable<int S::*>);

#define CHECK_POINTER_TO_MEMBER_FUNCTIONS(qualifier)                                                                   \
  static_assert(!std::weakly_incrementable<int (S::*)() qualifier>);                                                   \
  static_assert(!std::weakly_incrementable<int (S::*)() qualifier noexcept>);                                          \
  static_assert(!std::weakly_incrementable<int (S::*)() qualifier&>);                                                  \
  static_assert(!std::weakly_incrementable<int (S::*)() qualifier & noexcept>);                                        \
  static_assert(!std::weakly_incrementable<int (S::*)() qualifier&&>);                                                 \
  static_assert(!std::weakly_incrementable < int (S::*)() qualifier&& noexcept >);

#define NO_QUALIFIER
CHECK_POINTER_TO_MEMBER_FUNCTIONS(NO_QUALIFIER);
CHECK_POINTER_TO_MEMBER_FUNCTIONS(const);
CHECK_POINTER_TO_MEMBER_FUNCTIONS(volatile);
CHECK_POINTER_TO_MEMBER_FUNCTIONS(const volatile);

static_assert(std::weakly_incrementable<postfix_increment_returns_void>);
static_assert(std::weakly_incrementable<postfix_increment_returns_copy>);
static_assert(std::weakly_incrementable<has_integral_minus>);
static_assert(std::weakly_incrementable<has_distinct_difference_type_and_minus>);
static_assert(!std::weakly_incrementable<missing_difference_type>);
static_assert(!std::weakly_incrementable<floating_difference_type>);
static_assert(!std::weakly_incrementable<non_const_minus>);
static_assert(!std::weakly_incrementable<non_integral_minus>);
static_assert(!std::weakly_incrementable<bad_difference_type_good_minus>);
static_assert(!std::weakly_incrementable<not_movable>);
static_assert(!std::weakly_incrementable<preinc_not_declared>);
static_assert(!std::weakly_incrementable<postinc_not_declared>);
static_assert(std::weakly_incrementable<not_default_initializable>);
static_assert(std::weakly_incrementable<incrementable_with_difference_type>);
static_assert(std::weakly_incrementable<incrementable_without_difference_type>);
static_assert(std::weakly_incrementable<difference_type_and_void_minus>);
static_assert(std::weakly_incrementable<noncopyable_with_difference_type>);
static_assert(std::weakly_incrementable<noncopyable_without_difference_type>);
static_assert(std::weakly_incrementable<noncopyable_with_difference_type_and_minus>);
