//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <compare>

// template<class T, class U = T> struct compare_three_way_result;
// template<class T, class U = T>
//   using compare_three_way_result_t = typename compare_three_way_result<T, U>::type;

#include <compare>

#include "test_macros.h"

template<class T>
concept has_no_nested_type = !requires { typename T::type; };

ASSERT_SAME_TYPE(std::compare_three_way_result_t<int>, std::strong_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<float>, std::partial_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<unsigned>, std::strong_ordering);

ASSERT_SAME_TYPE(std::compare_three_way_result_t<int, int>, std::strong_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<int, float>, std::partial_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<float, int>, std::partial_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<float, float>, std::partial_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<float, unsigned>, std::partial_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<unsigned, float>, std::partial_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<unsigned, unsigned>, std::strong_ordering);

ASSERT_SAME_TYPE(std::compare_three_way_result_t<const int&>, std::strong_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<const int&, int>, std::strong_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<const int*>, std::strong_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<const int*, int*>, std::strong_ordering);

static_assert(has_no_nested_type<std::compare_three_way_result<void>>);
static_assert(has_no_nested_type<std::compare_three_way_result<void, void>>);
static_assert(has_no_nested_type<std::compare_three_way_result<int, void>>);
static_assert(has_no_nested_type<std::compare_three_way_result<int, int*>>);
static_assert(has_no_nested_type<std::compare_three_way_result<int, unsigned>>);
static_assert(has_no_nested_type<std::compare_three_way_result<unsigned, int>>);

struct A {
    float operator<=>(const A&) const;  // a non-comparison-category type is OK
};
ASSERT_SAME_TYPE(std::compare_three_way_result_t<A>, float);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<A, A>, float);

struct B {
    using T = int(&)();
    T operator<=>(const B&) const;  // no decay takes place either
};
ASSERT_SAME_TYPE(std::compare_three_way_result_t<B>, int(&)());
ASSERT_SAME_TYPE(std::compare_three_way_result_t<B, B>, int(&)());

struct C {
    std::strong_ordering operator<=>(C&);  // C isn't const-comparable
};
static_assert(has_no_nested_type<std::compare_three_way_result<C>>);
static_assert(has_no_nested_type<std::compare_three_way_result<C&>>);
static_assert(has_no_nested_type<std::compare_three_way_result<C&&>>);

static_assert(has_no_nested_type<std::compare_three_way_result<C, C>>);
static_assert(has_no_nested_type<std::compare_three_way_result<C&, C&>>);
static_assert(has_no_nested_type<std::compare_three_way_result<C&&, C&&>>);

struct D {
    std::strong_ordering operator<=>(D&) &;
    std::strong_ordering operator<=>(D&&) &&;
    std::weak_ordering operator<=>(const D&) const&;  // comparison is always done by const&
    std::strong_ordering operator<=>(const D&&) const&&;
};
ASSERT_SAME_TYPE(std::compare_three_way_result_t<D>, std::weak_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<D&>, std::weak_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<D&&>, std::weak_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<const D&>, std::weak_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<const D&&>, std::weak_ordering);

ASSERT_SAME_TYPE(std::compare_three_way_result_t<D, D>, std::weak_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<D&, D&>, std::weak_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<D&&, D&&>, std::weak_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<const D&, const D&>, std::weak_ordering);
ASSERT_SAME_TYPE(std::compare_three_way_result_t<const D&&, const D&&>, std::weak_ordering);
