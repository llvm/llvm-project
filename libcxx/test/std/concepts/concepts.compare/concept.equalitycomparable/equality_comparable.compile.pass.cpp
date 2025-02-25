//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T>
// concept equality_comparable = // see below

#include <concepts>

#include <array>
#include <cstddef>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "compare_types.h"

namespace fundamentals {
static_assert(std::equality_comparable<int>);
static_assert(std::equality_comparable<double>);
static_assert(std::equality_comparable<void*>);
static_assert(std::equality_comparable<char*>);
static_assert(std::equality_comparable<char const*>);
static_assert(std::equality_comparable<char volatile*>);
static_assert(std::equality_comparable<char const volatile*>);
static_assert(std::equality_comparable<wchar_t&>);
static_assert(std::equality_comparable<char8_t const&>);
static_assert(std::equality_comparable<char16_t volatile&>);
static_assert(std::equality_comparable<char32_t const volatile&>);
static_assert(std::equality_comparable<unsigned char&&>);
static_assert(std::equality_comparable<unsigned short const&&>);
static_assert(std::equality_comparable<unsigned int volatile&&>);
static_assert(std::equality_comparable<unsigned long const volatile&&>);
static_assert(std::equality_comparable<int[5]>);
static_assert(std::equality_comparable<int (*)(int)>);
static_assert(std::equality_comparable<int (&)(int)>);
static_assert(std::equality_comparable<int (*)(int) noexcept>);
static_assert(std::equality_comparable<int (&)(int) noexcept>);
static_assert(std::equality_comparable<std::nullptr_t>);

struct S {};
static_assert(std::equality_comparable<int S::*>);
static_assert(std::equality_comparable<int (S::*)()>);
static_assert(std::equality_comparable<int (S::*)() noexcept>);
static_assert(std::equality_comparable<int (S::*)() &>);
static_assert(std::equality_comparable<int (S::*)() & noexcept>);
static_assert(std::equality_comparable<int (S::*)() &&>);
static_assert(std::equality_comparable<int (S::*)() && noexcept>);
static_assert(std::equality_comparable<int (S::*)() const>);
static_assert(std::equality_comparable<int (S::*)() const noexcept>);
static_assert(std::equality_comparable<int (S::*)() const&>);
static_assert(std::equality_comparable<int (S::*)() const & noexcept>);
static_assert(std::equality_comparable<int (S::*)() const&&>);
static_assert(std::equality_comparable<int (S::*)() const && noexcept>);
static_assert(std::equality_comparable<int (S::*)() volatile>);
static_assert(std::equality_comparable<int (S::*)() volatile noexcept>);
static_assert(std::equality_comparable<int (S::*)() volatile&>);
static_assert(std::equality_comparable<int (S::*)() volatile & noexcept>);
static_assert(std::equality_comparable<int (S::*)() volatile&&>);
static_assert(std::equality_comparable<int (S::*)() volatile && noexcept>);
static_assert(std::equality_comparable<int (S::*)() const volatile>);
static_assert(std::equality_comparable<int (S::*)() const volatile noexcept>);
static_assert(std::equality_comparable<int (S::*)() const volatile&>);
static_assert(std::equality_comparable<int (S::*)() const volatile & noexcept>);
static_assert(std::equality_comparable<int (S::*)() const volatile&&>);
static_assert(
    std::equality_comparable<int (S::*)() const volatile && noexcept>);

static_assert(!std::equality_comparable<void>);
} // namespace fundamentals

namespace standard_types {
static_assert(std::equality_comparable<std::array<int, 10>>);
static_assert(std::equality_comparable<std::deque<int>>);
static_assert(std::equality_comparable<std::forward_list<int>>);
static_assert(std::equality_comparable<std::list<int>>);
static_assert(std::equality_comparable<std::map<int, void*>>);
static_assert(std::equality_comparable<std::optional<int>>);
static_assert(std::equality_comparable<std::set<int>>);
static_assert(std::equality_comparable<std::unordered_map<int, void*>>);
static_assert(std::equality_comparable<std::unordered_set<int>>);
static_assert(std::equality_comparable<std::vector<bool>>);
static_assert(std::equality_comparable<std::vector<int>>);
} // namespace standard_types

namespace types_fit_for_purpose {
static_assert(std::equality_comparable<cxx20_member_eq>);
static_assert(std::equality_comparable<cxx20_friend_eq>);
static_assert(std::equality_comparable<member_three_way_comparable>);
static_assert(std::equality_comparable<friend_three_way_comparable>);
static_assert(std::equality_comparable<explicit_operators>);
static_assert(std::equality_comparable<different_return_types>);
static_assert(std::equality_comparable<one_member_one_friend>);
static_assert(std::equality_comparable<equality_comparable_with_ec1>);

static_assert(!std::equality_comparable<no_eq>);
static_assert(!std::equality_comparable<no_neq>);
static_assert(std::equality_comparable<no_lt>);
static_assert(std::equality_comparable<no_gt>);
static_assert(std::equality_comparable<no_le>);
static_assert(std::equality_comparable<no_ge>);

static_assert(!std::equality_comparable<wrong_return_type_eq>);
static_assert(!std::equality_comparable<wrong_return_type_ne>);
static_assert(std::equality_comparable<wrong_return_type_lt>);
static_assert(std::equality_comparable<wrong_return_type_gt>);
static_assert(std::equality_comparable<wrong_return_type_le>);
static_assert(std::equality_comparable<wrong_return_type_ge>);
static_assert(!std::equality_comparable<wrong_return_type>);
static_assert(
    !std::equality_comparable<cxx20_member_eq_operator_with_deleted_ne>);
static_assert(
    !std::equality_comparable<cxx20_friend_eq_operator_with_deleted_ne>);
static_assert(
    !std::equality_comparable<member_three_way_comparable_with_deleted_eq>);
static_assert(
    !std::equality_comparable<member_three_way_comparable_with_deleted_ne>);
static_assert(
    !std::equality_comparable<friend_three_way_comparable_with_deleted_eq>);
static_assert(
    !std::equality_comparable<friend_three_way_comparable_with_deleted_ne>);

static_assert(!std::equality_comparable<eq_returns_explicit_bool>);
static_assert(!std::equality_comparable<ne_returns_explicit_bool>);
static_assert(std::equality_comparable<lt_returns_explicit_bool>);
static_assert(std::equality_comparable<gt_returns_explicit_bool>);
static_assert(std::equality_comparable<le_returns_explicit_bool>);
static_assert(std::equality_comparable<ge_returns_explicit_bool>);
static_assert(std::equality_comparable<returns_true_type>);
static_assert(std::equality_comparable<returns_int_ptr>);
} // namespace types_fit_for_purpose
