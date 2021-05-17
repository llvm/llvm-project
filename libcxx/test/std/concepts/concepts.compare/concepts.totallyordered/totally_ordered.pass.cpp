//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T>
// concept totally_ordered;

#include <concepts>

#include <array>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "compare_types.h"
#include "test_macros.h"

// `models_totally_ordered` checks that `std::totally_ordered` subsumes
// `std::equality_comparable`. This overload should *never* be called.
template <std::equality_comparable T>
constexpr bool models_totally_ordered() noexcept {
  return false;
}

template <std::totally_ordered T>
constexpr bool models_totally_ordered() noexcept {
  return true;
}

namespace fundamentals {
static_assert(models_totally_ordered<int>());
static_assert(models_totally_ordered<double>());
static_assert(models_totally_ordered<void*>());
static_assert(models_totally_ordered<char*>());
static_assert(models_totally_ordered<char const*>());
static_assert(models_totally_ordered<char volatile*>());
static_assert(models_totally_ordered<char const volatile*>());
static_assert(models_totally_ordered<wchar_t&>());
static_assert(models_totally_ordered<char8_t const&>());
static_assert(models_totally_ordered<char16_t volatile&>());
static_assert(models_totally_ordered<char32_t const volatile&>());
static_assert(models_totally_ordered<unsigned char&&>());
static_assert(models_totally_ordered<unsigned short const&&>());
static_assert(models_totally_ordered<unsigned int volatile&&>());
static_assert(models_totally_ordered<unsigned long const volatile&&>());
static_assert(models_totally_ordered<int[5]>());
static_assert(models_totally_ordered<int (*)(int)>());
static_assert(models_totally_ordered<int (&)(int)>());
static_assert(models_totally_ordered<int (*)(int) noexcept>());
static_assert(models_totally_ordered<int (&)(int) noexcept>());

#ifndef TEST_COMPILER_GCC
static_assert(!std::totally_ordered<std::nullptr_t>);
#endif

struct S {};
static_assert(!std::totally_ordered<S>);
static_assert(!std::totally_ordered<int S::*>);
static_assert(!std::totally_ordered<int (S::*)()>);
static_assert(!std::totally_ordered<int (S::*)() noexcept>);
static_assert(!std::totally_ordered<int (S::*)() &>);
static_assert(!std::totally_ordered<int (S::*)() & noexcept>);
static_assert(!std::totally_ordered<int (S::*)() &&>);
static_assert(!std::totally_ordered < int (S::*)() && noexcept >);
static_assert(!std::totally_ordered<int (S::*)() const>);
static_assert(!std::totally_ordered<int (S::*)() const noexcept>);
static_assert(!std::totally_ordered<int (S::*)() const&>);
static_assert(!std::totally_ordered<int (S::*)() const & noexcept>);
static_assert(!std::totally_ordered<int (S::*)() const&&>);
static_assert(!std::totally_ordered < int (S::*)() const&& noexcept >);
static_assert(!std::totally_ordered<int (S::*)() volatile>);
static_assert(!std::totally_ordered<int (S::*)() volatile noexcept>);
static_assert(!std::totally_ordered<int (S::*)() volatile&>);
static_assert(!std::totally_ordered<int (S::*)() volatile & noexcept>);
static_assert(!std::totally_ordered<int (S::*)() volatile&&>);
static_assert(!std::totally_ordered < int (S::*)() volatile&& noexcept >);
static_assert(!std::totally_ordered<int (S::*)() const volatile>);
static_assert(!std::totally_ordered<int (S::*)() const volatile noexcept>);
static_assert(!std::totally_ordered<int (S::*)() const volatile&>);
static_assert(!std::totally_ordered<int (S::*)() const volatile & noexcept>);
static_assert(!std::totally_ordered<int (S::*)() const volatile&&>);
static_assert(!std::totally_ordered < int (S::*)() const volatile&& noexcept >);

static_assert(!std::totally_ordered<void>);
} // namespace fundamentals

namespace standard_types {
static_assert(models_totally_ordered<std::array<int, 10> >());
static_assert(models_totally_ordered<std::deque<int> >());
static_assert(models_totally_ordered<std::forward_list<int> >());
static_assert(models_totally_ordered<std::list<int> >());
static_assert(models_totally_ordered<std::optional<int> >());
static_assert(models_totally_ordered<std::set<int> >());
static_assert(models_totally_ordered<std::vector<bool> >());
static_assert(models_totally_ordered<std::vector<int> >());

static_assert(!std::totally_ordered<std::unordered_map<int, void*> >);
static_assert(!std::totally_ordered<std::unordered_set<int> >);

struct A {};
// FIXME(cjdb): uncomment when operator<=> is implemented for each of these types.
// static_assert(!std::totally_ordered<std::array<A, 10> >);
// static_assert(!std::totally_ordered<std::deque<A> >);
// static_assert(!std::totally_ordered<std::forward_list<A> >);
// static_assert(!std::totally_ordered<std::list<A> >);
static_assert(!std::totally_ordered<std::optional<A> >);
// static_assert(!std::totally_ordered<std::set<A> >);
// static_assert(!std::totally_ordered<std::vector<A> >);
} // namespace standard_types

namespace types_fit_for_purpose {
static_assert(models_totally_ordered<member_three_way_comparable>());
static_assert(models_totally_ordered<friend_three_way_comparable>());
static_assert(models_totally_ordered<explicit_operators>());
static_assert(models_totally_ordered<different_return_types>());
static_assert(!std::totally_ordered<cxx20_member_eq>);
static_assert(!std::totally_ordered<cxx20_friend_eq>);
static_assert(!std::totally_ordered<one_member_one_friend>);
static_assert(!std::totally_ordered<equality_comparable_with_ec1>);

static_assert(!std::totally_ordered<no_eq>);
static_assert(!std::totally_ordered<no_neq>);
static_assert(!std::totally_ordered<no_lt>);
static_assert(!std::totally_ordered<no_gt>);
static_assert(!std::totally_ordered<no_le>);
static_assert(!std::totally_ordered<no_ge>);

static_assert(!std::totally_ordered<wrong_return_type_eq>);
static_assert(!std::totally_ordered<wrong_return_type_ne>);
static_assert(!std::totally_ordered<wrong_return_type_lt>);
static_assert(!std::totally_ordered<wrong_return_type_gt>);
static_assert(!std::totally_ordered<wrong_return_type_le>);
static_assert(!std::totally_ordered<wrong_return_type_ge>);
static_assert(!std::totally_ordered<wrong_return_type>);

static_assert(!std::totally_ordered<cxx20_member_eq_operator_with_deleted_ne>);
static_assert(!std::totally_ordered<cxx20_friend_eq_operator_with_deleted_ne>);
static_assert(
    !std::totally_ordered<member_three_way_comparable_with_deleted_eq>);
static_assert(
    !std::totally_ordered<member_three_way_comparable_with_deleted_ne>);
static_assert(
    !std::totally_ordered<friend_three_way_comparable_with_deleted_eq>);
static_assert(
    !std::totally_ordered<friend_three_way_comparable_with_deleted_ne>);

static_assert(!std::totally_ordered<eq_returns_explicit_bool>);
static_assert(!std::totally_ordered<ne_returns_explicit_bool>);
static_assert(!std::totally_ordered<lt_returns_explicit_bool>);
static_assert(!std::totally_ordered<gt_returns_explicit_bool>);
static_assert(!std::totally_ordered<le_returns_explicit_bool>);
static_assert(!std::totally_ordered<ge_returns_explicit_bool>);
static_assert(std::totally_ordered<returns_true_type>);
static_assert(std::totally_ordered<returns_int_ptr>);

static_assert(std::totally_ordered<partial_ordering_totally_ordered_with>);
static_assert(std::totally_ordered<weak_ordering_totally_ordered_with>);
static_assert(std::totally_ordered<strong_ordering_totally_ordered_with>);
} // namespace types_fit_for_purpose

int main(int, char**) { return 0; }
