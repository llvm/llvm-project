//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T, class U, class Cat = partial_ordering>
// concept three_way_comparable_with = // see below

#include <compare>

#include "compare_types.h"

template <class T, class U = T, typename Cat = std::partial_ordering>
constexpr bool check_three_way_comparable_with() {
  constexpr bool result = std::three_way_comparable_with<T, U, Cat>;
  static_assert(std::three_way_comparable_with<U, T, Cat> == result);
  static_assert(std::three_way_comparable_with<T, U const, Cat> == result);
  static_assert(std::three_way_comparable_with<T const, U const, Cat> == result);
  static_assert(std::three_way_comparable_with<T, U const&, Cat> == result);
  static_assert(std::three_way_comparable_with<T const, U const&, Cat> == result);
  static_assert(std::three_way_comparable_with<T, U const&&, Cat> == result);
  static_assert(std::three_way_comparable_with<T const, U const&&, Cat> == result);
  if constexpr (!std::is_void_v<T>) {
    static_assert(std::three_way_comparable_with<T&, U const, Cat> == result);
    static_assert(std::three_way_comparable_with<T const&, U const, Cat> == result);
    static_assert(std::three_way_comparable_with<T&, U const&, Cat> == result);
    static_assert(std::three_way_comparable_with<T const&, U const&, Cat> == result);
    static_assert(std::three_way_comparable_with<T&, U const&&, Cat> == result);
    static_assert(std::three_way_comparable_with<T const&, U const&&, Cat> == result);
    static_assert(std::three_way_comparable_with<T&&, U const, Cat> == result);
    static_assert(std::three_way_comparable_with<T const&&, U const, Cat> == result);
    static_assert(std::three_way_comparable_with<T&&, U const&, Cat> == result);
    static_assert(std::three_way_comparable_with<T const&&, U const&, Cat> == result);
    static_assert(std::three_way_comparable_with<T&&, U const&&, Cat> == result);
    static_assert(std::three_way_comparable_with<T const&&, U const&&, Cat> == result);
  }
  return result;
}

namespace fundamentals {
static_assert(check_three_way_comparable_with<int, int>());
static_assert(check_three_way_comparable_with<int, char>());
static_assert(!check_three_way_comparable_with<int, unsigned int>());
static_assert(check_three_way_comparable_with<int, double>());
static_assert(check_three_way_comparable_with<int*, int*>());

static_assert(check_three_way_comparable_with<int, int, std::strong_ordering>());
static_assert(check_three_way_comparable_with<int, char, std::strong_ordering>());
static_assert(check_three_way_comparable_with<int, short, std::strong_ordering>());

static_assert(check_three_way_comparable_with<int, int, std::weak_ordering>());
static_assert(check_three_way_comparable_with<int, char, std::weak_ordering>());
static_assert(!check_three_way_comparable_with<int, unsigned int, std::weak_ordering>());

static_assert(!check_three_way_comparable_with<int, bool>());
static_assert(!check_three_way_comparable_with<int, int*>());
static_assert(!check_three_way_comparable_with<int, int[5]>());
static_assert(!check_three_way_comparable_with<int, int (*)()>());
static_assert(!check_three_way_comparable_with<int, int (&)()>());
struct S {};
static_assert(!check_three_way_comparable_with<int, int S::*>());
static_assert(!check_three_way_comparable_with<int, int (S::*)()>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() noexcept>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() const>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() const noexcept>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() volatile>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() volatile noexcept>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() const volatile>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() const volatile noexcept>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() &>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() & noexcept>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() const&>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() const & noexcept>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() volatile&>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() volatile & noexcept>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() const volatile&>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() const volatile & noexcept>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() &&>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() && noexcept>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() const&&>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() const&& noexcept>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() volatile&&>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() volatile&& noexcept>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() const volatile&&>());
static_assert(!check_three_way_comparable_with<int, int (S::*)() const volatile&& noexcept>());
static_assert(!check_three_way_comparable_with<int*, int[5]>());
static_assert(!check_three_way_comparable_with<int[5], int[5]>());
static_assert(!check_three_way_comparable_with<std::nullptr_t, int>());
static_assert(!check_three_way_comparable_with<std::nullptr_t, int*>());
static_assert(!check_three_way_comparable_with<std::nullptr_t, int[5]>());
static_assert(!check_three_way_comparable_with<std::nullptr_t, int (*)()>());
static_assert(!check_three_way_comparable_with<std::nullptr_t, int (&)()>());
static_assert(!check_three_way_comparable_with<std::nullptr_t, int (S::*)()>());
static_assert(!check_three_way_comparable_with<void, int>());
static_assert(!check_three_way_comparable_with<void, int*>());
static_assert(!check_three_way_comparable_with<void, std::nullptr_t>());
static_assert(!check_three_way_comparable_with<void, int[5]>());
static_assert(!check_three_way_comparable_with<void, int (*)()>());
static_assert(!check_three_way_comparable_with<void, int (&)()>());
static_assert(!check_three_way_comparable_with<void, int S::*>());
static_assert(!check_three_way_comparable_with<void, int (S::*)()>());
} // namespace fundamentals

namespace user_defined {
struct S {
    bool operator==(int) const;
    std::strong_ordering operator<=>(int) const;
    operator int() const;

    bool operator==(const S&) const = default;
    auto operator<=>(const S&) const = default;
};

static_assert(check_three_way_comparable_with<S, int>());
static_assert(check_three_way_comparable_with<S, int, std::strong_ordering>());
static_assert(check_three_way_comparable_with<S, int, std::weak_ordering>());

struct SpaceshipNotDeclared {
};

static_assert(!check_three_way_comparable_with<SpaceshipNotDeclared>());

struct SpaceshipDeleted {
    auto operator<=>(const SpaceshipDeleted&) const = delete;
};

static_assert(!check_three_way_comparable_with<SpaceshipDeleted>());

struct SpaceshipWithoutEqualityOperator {
    auto operator<=>(const SpaceshipWithoutEqualityOperator&) const;
};

static_assert(!check_three_way_comparable_with<SpaceshipWithoutEqualityOperator>());

struct EqualityOperatorDeleted {
    bool operator==(const EqualityOperatorDeleted&) const = delete;
};

static_assert(!check_three_way_comparable_with<EqualityOperatorDeleted>());

struct EqualityOperatorOnly {
    bool operator==(const EqualityOperatorOnly&) const = default;
};

static_assert(!check_three_way_comparable_with<EqualityOperatorOnly>());

struct SpaceshipDeclaredEqualityOperatorDeleted {
    bool operator==(const SpaceshipDeclaredEqualityOperatorDeleted&) const = delete;
    auto operator<=>(const SpaceshipDeclaredEqualityOperatorDeleted&) const = default;
};

static_assert(!check_three_way_comparable_with<SpaceshipDeclaredEqualityOperatorDeleted>());

struct AllInequalityOperators {
    bool operator<(const AllInequalityOperators&) const;
    bool operator<=(const AllInequalityOperators&) const;
    bool operator>(const AllInequalityOperators&) const;
    bool operator>=(const AllInequalityOperators&) const;
    bool operator!=(const AllInequalityOperators&) const;
};

static_assert(!check_three_way_comparable_with<AllInequalityOperators>());

struct AllComparisonOperators {
    bool operator<(const AllComparisonOperators&) const;
    bool operator<=(const AllComparisonOperators&) const;
    bool operator>(const AllComparisonOperators&) const;
    bool operator>=(const AllComparisonOperators&) const;
    bool operator!=(const AllComparisonOperators&) const;
    bool operator==(const AllComparisonOperators&) const;
};

static_assert(!check_three_way_comparable_with<AllComparisonOperators>());

struct AllButOneInequalityOperators {
    bool operator<(const AllButOneInequalityOperators&) const;
    bool operator<=(const AllButOneInequalityOperators&) const;
    bool operator>(const AllButOneInequalityOperators&) const;
    bool operator!=(const AllButOneInequalityOperators&) const;
};

static_assert(!check_three_way_comparable_with<AllButOneInequalityOperators>());

struct AllInequalityOperatorsOneDeleted {
    bool operator<(const AllInequalityOperatorsOneDeleted&) const;
    bool operator<=(const AllInequalityOperatorsOneDeleted&) const;
    bool operator>(const AllInequalityOperatorsOneDeleted&) const;
    bool operator>=(const AllInequalityOperatorsOneDeleted&) const = delete;
    bool operator!=(const AllInequalityOperatorsOneDeleted&) const;
};

static_assert(!check_three_way_comparable_with<AllInequalityOperatorsOneDeleted>());

struct EqualityOperatorWrongReturnType {
    int operator==(const EqualityOperatorWrongReturnType&);
    auto operator<=>(const EqualityOperatorWrongReturnType&) const = default;
};

static_assert(!check_three_way_comparable_with<EqualityOperatorWrongReturnType>());

struct SpaceshipWrongReturnType {
    bool operator==(const SpaceshipWrongReturnType&) const = default;
    int operator<=>(const SpaceshipWrongReturnType&);
};

static_assert(!check_three_way_comparable_with<SpaceshipWrongReturnType>());

struct EqualityOperatorNonConstArgument {
    bool operator==(EqualityOperatorNonConstArgument&);
    auto operator<=>(const EqualityOperatorNonConstArgument&) const = default;
};

static_assert(!check_three_way_comparable_with<EqualityOperatorNonConstArgument>());

struct SpaceshipNonConstArgument {
    bool operator==(const SpaceshipNonConstArgument&) const = default;
    auto operator<=>(SpaceshipNonConstArgument&);
};

static_assert(!check_three_way_comparable_with<SpaceshipNonConstArgument>());
} // namespace user_defined
