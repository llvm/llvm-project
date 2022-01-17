//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class LHS, class RHS>
// concept assignable_from =
//   std::is_lvalue_reference_v<LHS> &&
//   std::common_reference_with<
//     const std::remove_reference_t<LHS>&,
//     const std::remove_reference_t<RHS>&> &&
//   requires (LHS lhs, RHS&& rhs) {
//     { lhs = std::forward<RHS>(rhs) } -> std::same_as<LHS>;
//   };

#include <concepts>
#include <type_traits>

#include "MoveOnly.h"

struct NoCommonRef {
  NoCommonRef& operator=(const int&);
};
static_assert(std::is_assignable_v<NoCommonRef&, const int&>);
static_assert(!std::assignable_from<NoCommonRef&, const int&>); // no common reference type

struct Base {};
struct Derived : Base {};
static_assert(!std::assignable_from<Base*, Derived*>);
static_assert( std::assignable_from<Base*&, Derived*>);
static_assert( std::assignable_from<Base*&, Derived*&>);
static_assert( std::assignable_from<Base*&, Derived*&&>);
static_assert( std::assignable_from<Base*&, Derived* const>);
static_assert( std::assignable_from<Base*&, Derived* const&>);
static_assert( std::assignable_from<Base*&, Derived* const&&>);
static_assert(!std::assignable_from<Base*&, const Derived*>);
static_assert(!std::assignable_from<Base*&, const Derived*&>);
static_assert(!std::assignable_from<Base*&, const Derived*&&>);
static_assert(!std::assignable_from<Base*&, const Derived* const>);
static_assert(!std::assignable_from<Base*&, const Derived* const&>);
static_assert(!std::assignable_from<Base*&, const Derived* const&&>);
static_assert( std::assignable_from<const Base*&, Derived*>);
static_assert( std::assignable_from<const Base*&, Derived*&>);
static_assert( std::assignable_from<const Base*&, Derived*&&>);
static_assert( std::assignable_from<const Base*&, Derived* const>);
static_assert( std::assignable_from<const Base*&, Derived* const&>);
static_assert( std::assignable_from<const Base*&, Derived* const&&>);
static_assert( std::assignable_from<const Base*&, const Derived*>);
static_assert( std::assignable_from<const Base*&, const Derived*&>);
static_assert( std::assignable_from<const Base*&, const Derived*&&>);
static_assert( std::assignable_from<const Base*&, const Derived* const>);
static_assert( std::assignable_from<const Base*&, const Derived* const&>);
static_assert( std::assignable_from<const Base*&, const Derived* const&&>);

struct VoidResultType {
    void operator=(const VoidResultType&);
};
static_assert(std::is_assignable_v<VoidResultType&, const VoidResultType&>);
static_assert(!std::assignable_from<VoidResultType&, const VoidResultType&>);

struct ValueResultType {
    ValueResultType operator=(const ValueResultType&);
};
static_assert(std::is_assignable_v<ValueResultType&, const ValueResultType&>);
static_assert(!std::assignable_from<ValueResultType&, const ValueResultType&>);

struct Locale {
    const Locale& operator=(const Locale&);
};
static_assert(std::is_assignable_v<Locale&, const Locale&>);
static_assert(!std::assignable_from<Locale&, const Locale&>);

struct Tuple {
    Tuple& operator=(const Tuple&);
    const Tuple& operator=(const Tuple&) const;
};
static_assert(!std::assignable_from<Tuple, const Tuple&>);
static_assert( std::assignable_from<Tuple&, const Tuple&>);
static_assert(!std::assignable_from<Tuple&&, const Tuple&>);
static_assert(!std::assignable_from<const Tuple, const Tuple&>);
static_assert( std::assignable_from<const Tuple&, const Tuple&>);
static_assert(!std::assignable_from<const Tuple&&, const Tuple&>);

// Finally, check a few simple cases.
static_assert( std::assignable_from<int&, int>);
static_assert( std::assignable_from<int&, int&>);
static_assert( std::assignable_from<int&, int&&>);
static_assert(!std::assignable_from<const int&, int>);
static_assert(!std::assignable_from<const int&, int&>);
static_assert(!std::assignable_from<const int&, int&&>);
static_assert( std::assignable_from<volatile int&, int>);
static_assert( std::assignable_from<volatile int&, int&>);
static_assert( std::assignable_from<volatile int&, int&&>);
static_assert(!std::assignable_from<int(&)[10], int>);
static_assert(!std::assignable_from<int(&)[10], int(&)[10]>);
static_assert( std::assignable_from<MoveOnly&, MoveOnly>);
static_assert(!std::assignable_from<MoveOnly&, MoveOnly&>);
static_assert( std::assignable_from<MoveOnly&, MoveOnly&&>);
static_assert(!std::assignable_from<void, int>);
static_assert(!std::assignable_from<void, void>);
