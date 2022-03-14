//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T, class Cat = partial_ordering>
// concept three_way_comparable = // see below

#include <compare>

#include "compare_types.h"
#include "test_macros.h"

namespace fundamentals {
// with default ordering
static_assert(std::three_way_comparable<int>);
static_assert(std::three_way_comparable<double>);
static_assert(std::three_way_comparable<void*>);
static_assert(std::three_way_comparable<char*>);
static_assert(std::three_way_comparable<char const*>);
static_assert(std::three_way_comparable<char volatile*>);
static_assert(std::three_way_comparable<char const volatile*>);
static_assert(std::three_way_comparable<wchar_t&>);
#ifndef TEST_HAS_NO_CHAR8_T
static_assert(std::three_way_comparable<char8_t const&>);
#endif
#ifndef TEST_HAS_NO_UNICODE_CHARS
static_assert(std::three_way_comparable<char16_t volatile&>);
static_assert(std::three_way_comparable<char32_t const volatile&>);
#endif
#ifndef TEST_HAS_NO_INT128
static_assert(std::three_way_comparable<__int128_t const&>);
static_assert(std::three_way_comparable<__uint128_t const&>);
#endif
static_assert(std::three_way_comparable<unsigned char&&>);
static_assert(std::three_way_comparable<unsigned short const&&>);
static_assert(std::three_way_comparable<unsigned int volatile&&>);
static_assert(std::three_way_comparable<unsigned long const volatile&&>);

// with explicit ordering
static_assert(std::three_way_comparable<int, std::strong_ordering>);
static_assert(std::three_way_comparable<int, std::weak_ordering>);
static_assert(std::three_way_comparable<double, std::partial_ordering>);
static_assert(!std::three_way_comparable<double, std::weak_ordering>);
static_assert(std::three_way_comparable<void*, std::strong_ordering>);
static_assert(std::three_way_comparable<void*, std::weak_ordering>);
static_assert(std::three_way_comparable<char*, std::strong_ordering>);
static_assert(std::three_way_comparable<char*, std::weak_ordering>);
static_assert(std::three_way_comparable<char const*, std::strong_ordering>);
static_assert(std::three_way_comparable<char const*, std::weak_ordering>);
static_assert(std::three_way_comparable<char volatile*, std::strong_ordering>);
static_assert(std::three_way_comparable<char volatile*, std::weak_ordering>);
static_assert(std::three_way_comparable<char const volatile*, std::strong_ordering>);
static_assert(std::three_way_comparable<char const volatile*, std::weak_ordering>);
static_assert(std::three_way_comparable<wchar_t&, std::strong_ordering>);
static_assert(std::three_way_comparable<wchar_t&, std::weak_ordering>);
static_assert(std::three_way_comparable<char8_t const&, std::strong_ordering>);
static_assert(std::three_way_comparable<char8_t const&, std::weak_ordering>);
static_assert(std::three_way_comparable<char16_t volatile&, std::strong_ordering>);
static_assert(std::three_way_comparable<char16_t volatile&, std::weak_ordering>);
static_assert(std::three_way_comparable<char32_t const volatile&, std::strong_ordering>);
static_assert(std::three_way_comparable<char32_t const volatile&, std::weak_ordering>);
static_assert(std::three_way_comparable<unsigned char&&, std::strong_ordering>);
static_assert(std::three_way_comparable<unsigned char&&, std::weak_ordering>);
static_assert(std::three_way_comparable<unsigned short const&&, std::strong_ordering>);
static_assert(std::three_way_comparable<unsigned short const&&, std::weak_ordering>);
static_assert(std::three_way_comparable<unsigned int volatile&&, std::strong_ordering>);
static_assert(std::three_way_comparable<unsigned int volatile&&, std::weak_ordering>);
static_assert(std::three_way_comparable<unsigned long const volatile&&, std::strong_ordering>);
static_assert(std::three_way_comparable<unsigned long const volatile&&, std::weak_ordering>);

static_assert(!std::three_way_comparable<int[5]>);
static_assert(!std::three_way_comparable<int (*)(int)>);
static_assert(!std::three_way_comparable<int (&)(int)>);
static_assert(!std::three_way_comparable<int (*)(int) noexcept>);
static_assert(!std::three_way_comparable<int (&)(int) noexcept>);
static_assert(!std::three_way_comparable<std::nullptr_t>);
static_assert(!std::three_way_comparable<void>);

struct S {};
static_assert(!std::three_way_comparable<int S::*>);
static_assert(!std::three_way_comparable<int (S::*)()>);
static_assert(!std::three_way_comparable<int (S::*)() noexcept>);
static_assert(!std::three_way_comparable<int (S::*)() &>);
static_assert(!std::three_way_comparable<int (S::*)() & noexcept>);
static_assert(!std::three_way_comparable<int (S::*)() &&>);
static_assert(!std::three_way_comparable<int (S::*)() && noexcept>);
static_assert(!std::three_way_comparable<int (S::*)() const>);
static_assert(!std::three_way_comparable<int (S::*)() const noexcept>);
static_assert(!std::three_way_comparable<int (S::*)() const&>);
static_assert(!std::three_way_comparable<int (S::*)() const & noexcept>);
static_assert(!std::three_way_comparable<int (S::*)() const&&>);
static_assert(!std::three_way_comparable<int (S::*)() const && noexcept>);
static_assert(!std::three_way_comparable<int (S::*)() volatile>);
static_assert(!std::three_way_comparable<int (S::*)() volatile noexcept>);
static_assert(!std::three_way_comparable<int (S::*)() volatile&>);
static_assert(!std::three_way_comparable<int (S::*)() volatile & noexcept>);
static_assert(!std::three_way_comparable<int (S::*)() volatile&&>);
static_assert(!std::three_way_comparable<int (S::*)() volatile && noexcept>);
static_assert(!std::three_way_comparable<int (S::*)() const volatile>);
static_assert(!std::three_way_comparable<int (S::*)() const volatile noexcept>);
static_assert(!std::three_way_comparable<int (S::*)() const volatile&>);
static_assert(!std::three_way_comparable<int (S::*)() const volatile & noexcept>);
static_assert(!std::three_way_comparable<int (S::*)() const volatile&&>);
static_assert(!std::three_way_comparable<int (S::*)() const volatile && noexcept>);
} // namespace fundamentals

namespace user_defined {

struct S {
    auto operator<=>(const S&) const = default;
};

static_assert(std::three_way_comparable<S>);
static_assert(std::three_way_comparable<S, std::strong_ordering>);
static_assert(std::three_way_comparable<S, std::partial_ordering>);

struct SpaceshipNotDeclared {
};

static_assert(!std::three_way_comparable<SpaceshipNotDeclared>);

struct SpaceshipDeleted {
    auto operator<=>(const SpaceshipDeleted&) const = delete;
};

static_assert(!std::three_way_comparable<SpaceshipDeleted>);

struct SpaceshipWithoutEqualityOperator {
    auto operator<=>(const SpaceshipWithoutEqualityOperator&) const;
};

static_assert(!std::three_way_comparable<SpaceshipWithoutEqualityOperator>);

struct EqualityOperatorDeleted {
    bool operator==(const EqualityOperatorDeleted&) const = delete;
};

static_assert(!std::three_way_comparable<EqualityOperatorDeleted>);

struct EqualityOperatorOnly {
    bool operator==(const EqualityOperatorOnly&) const = default;
};

static_assert(!std::three_way_comparable<EqualityOperatorOnly>);

struct SpaceshipDeclaredEqualityOperatorDeleted {
    bool operator==(const SpaceshipDeclaredEqualityOperatorDeleted&) const = delete;
    auto operator<=>(const SpaceshipDeclaredEqualityOperatorDeleted&) const = default;
};

static_assert(!std::three_way_comparable<SpaceshipDeclaredEqualityOperatorDeleted>);

struct AllInequalityOperators {
    bool operator<(const AllInequalityOperators&) const;
    bool operator<=(const AllInequalityOperators&) const;
    bool operator>(const AllInequalityOperators&) const;
    bool operator>=(const AllInequalityOperators&) const;
    bool operator!=(const AllInequalityOperators&) const;
};

static_assert(!std::three_way_comparable<AllInequalityOperators>);

struct AllComparisonOperators {
    bool operator<(const AllComparisonOperators&) const;
    bool operator<=(const AllComparisonOperators&) const;
    bool operator>(const AllComparisonOperators&) const;
    bool operator>=(const AllComparisonOperators&) const;
    bool operator!=(const AllComparisonOperators&) const;
    bool operator==(const AllComparisonOperators&) const;
};

static_assert(!std::three_way_comparable<AllComparisonOperators>);

struct AllButOneInequalityOperators {
    bool operator<(const AllButOneInequalityOperators&) const;
    bool operator<=(const AllButOneInequalityOperators&) const;
    bool operator>(const AllButOneInequalityOperators&) const;
    bool operator!=(const AllButOneInequalityOperators&) const;
};

static_assert(!std::three_way_comparable<AllButOneInequalityOperators>);

struct AllInequalityOperatorsOneDeleted {
    bool operator<(const AllInequalityOperatorsOneDeleted&) const;
    bool operator<=(const AllInequalityOperatorsOneDeleted&) const;
    bool operator>(const AllInequalityOperatorsOneDeleted&) const;
    bool operator>=(const AllInequalityOperatorsOneDeleted&) const = delete;
    bool operator!=(const AllInequalityOperatorsOneDeleted&) const;
};

static_assert(!std::three_way_comparable<AllInequalityOperatorsOneDeleted>);

struct EqualityOperatorWrongReturnType {
    int operator==(const EqualityOperatorWrongReturnType&);
    auto operator<=>(const EqualityOperatorWrongReturnType&) const = default;
};

static_assert(!std::three_way_comparable<EqualityOperatorWrongReturnType>);

struct SpaceshipWrongReturnType {
    bool operator==(const SpaceshipWrongReturnType&) const = default;
    int operator<=>(const SpaceshipWrongReturnType&);
};

static_assert(!std::three_way_comparable<SpaceshipWrongReturnType>);

struct EqualityOperatorNonConstArgument {
    bool operator==(EqualityOperatorNonConstArgument&);
    auto operator<=>(const EqualityOperatorNonConstArgument&) const = default;
};

static_assert(!std::three_way_comparable<EqualityOperatorNonConstArgument>);

struct SpaceshipNonConstArgument {
    bool operator==(const SpaceshipNonConstArgument&) const = default;
    auto operator<=>(SpaceshipNonConstArgument&);
};

static_assert(!std::three_way_comparable<SpaceshipNonConstArgument>);
} // namespace user_defined
