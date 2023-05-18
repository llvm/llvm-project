//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <compare>

// template<class T> constexpr strong_ordering compare_strong_order_fallback(const T& a, const T& b);

#include <compare>

#include <cassert>
#include <cmath>
#include <iterator> // std::size
#include <limits>
#include <type_traits>
#include <utility>

#include "test_macros.h"

#if defined(__i386__)
 #define TEST_BUGGY_SIGNALING_NAN
#endif

template<class T, class U>
constexpr auto has_strong_order(T&& t, U&& u)
    -> decltype(std::compare_strong_order_fallback(static_cast<T&&>(t), static_cast<U&&>(u)), true)
{
    return true;
}

constexpr bool has_strong_order(...) {
    return false;
}

namespace N11 {
    struct A {};
    struct B {};
    std::strong_ordering strong_order(const A&, const A&) { return std::strong_ordering::less; }
    std::strong_ordering strong_order(const A&, const B&);
}

void test_1_1()
{
    // If the decayed types of E and F differ, strong_order(E, F) is ill-formed.

    static_assert( has_strong_order(1, 2));
    static_assert(!has_strong_order(1, (short)2));
    static_assert(!has_strong_order(1, 2.0));
    static_assert(!has_strong_order(1.0f, 2.0));

    static_assert( has_strong_order((int*)nullptr, (int*)nullptr));
    static_assert(!has_strong_order((int*)nullptr, (const int*)nullptr));
    static_assert(!has_strong_order((const int*)nullptr, (int*)nullptr));
    static_assert( has_strong_order((const int*)nullptr, (const int*)nullptr));

    N11::A a;
    N11::B b;
    static_assert( has_strong_order(a, a));
    static_assert(!has_strong_order(a, b));
}

namespace N12 {
    struct A {};
    std::strong_ordering strong_order(A&, A&&) { return std::strong_ordering::less; }
    std::strong_ordering strong_order(A&&, A&&) { return std::strong_ordering::equal; }
    std::strong_ordering strong_order(const A&, const A&);

    struct B {
        friend std::weak_ordering strong_order(B&, B&);
    };

    struct StrongOrder {
        explicit operator std::strong_ordering() const { return std::strong_ordering::less; }
    };
    struct C {
        bool touched = false;
        friend StrongOrder strong_order(C& lhs, C&) { lhs.touched = true; return StrongOrder(); }
    };
}

void test_1_2()
{
    // Otherwise, strong_ordering(strong_order(E, F))
    // if it is a well-formed expression with overload resolution performed
    // in a context that does not include a declaration of std::strong_order.

    // Test that strong_order does not const-qualify the forwarded arguments.
    N12::A a;
    assert(std::compare_strong_order_fallback(a, std::move(a)) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(std::move(a), std::move(a)) == std::strong_ordering::equal);

    // The type of strong_order(e,f) must be explicitly convertible to strong_ordering.
    N12::B b;
    static_assert(!has_strong_order(b, b));

    N12::C c1, c2;
    ASSERT_SAME_TYPE(decltype(std::compare_strong_order_fallback(c1, c2)), std::strong_ordering);
    assert(std::compare_strong_order_fallback(c1, c2) == std::strong_ordering::less);
    assert(c1.touched);
    assert(!c2.touched);
}

template<class F>
constexpr bool test_1_3()
{
    // Otherwise, if the decayed type T of E is a floating-point type,
    // yields a value of type strong_ordering that is consistent with
    // the ordering observed by T's comparison operators,
    // and if numeric_limits<T>::is_iec559 is true, is additionally consistent with
    // the totalOrder operation as specified in ISO/IEC/IEEE 60559.

    static_assert(std::numeric_limits<F>::is_iec559);

    ASSERT_SAME_TYPE(decltype(std::compare_strong_order_fallback(F(0), F(0))), std::strong_ordering);

    F v[] = {
        -std::numeric_limits<F>::infinity(),
        std::numeric_limits<F>::lowest(),  // largest (finite) negative number
        F(-1.0), F(-0.1),
        -std::numeric_limits<F>::min(),    // smallest (normal) negative number
        F(-0.0),                           // negative zero
        F(0.0),
        std::numeric_limits<F>::min(),     // smallest (normal) positive number
        F(0.1), F(1.0), F(2.0), F(3.14),
        std::numeric_limits<F>::max(),     // largest (finite) positive number
        std::numeric_limits<F>::infinity(),
    };

    static_assert(std::size(v) == 14);

    // Sanity-check that array 'v' is indeed in the right order.
    for (int i=0; i < 14; ++i) {
        for (int j=0; j < 14; ++j) {
            auto naturalOrder = (v[i] <=> v[j]);
            if (v[i] == 0 && v[j] == 0) {
                assert(naturalOrder == std::partial_ordering::equivalent);
            } else {
                assert(naturalOrder == std::partial_ordering::unordered || naturalOrder == (i <=> j));
            }
        }
    }

    assert(std::compare_strong_order_fallback(v[0], v[0]) == std::strong_ordering::equal);
    assert(std::compare_strong_order_fallback(v[0], v[1]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[0], v[2]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[0], v[3]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[0], v[4]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[0], v[5]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[0], v[6]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[0], v[7]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[0], v[8]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[0], v[9]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[0], v[10]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[0], v[11]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[0], v[12]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[0], v[13]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[1], v[0]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[1], v[1]) == std::strong_ordering::equal);
    assert(std::compare_strong_order_fallback(v[1], v[2]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[1], v[3]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[1], v[4]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[1], v[5]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[1], v[6]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[1], v[7]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[1], v[8]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[1], v[9]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[1], v[10]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[1], v[11]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[1], v[12]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[1], v[13]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[2], v[0]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[2], v[1]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[2], v[2]) == std::strong_ordering::equal);
    assert(std::compare_strong_order_fallback(v[2], v[3]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[2], v[4]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[2], v[5]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[2], v[6]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[2], v[7]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[2], v[8]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[2], v[9]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[2], v[10]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[2], v[11]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[2], v[12]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[2], v[13]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[3], v[0]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[3], v[1]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[3], v[2]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[3], v[3]) == std::strong_ordering::equal);
    assert(std::compare_strong_order_fallback(v[3], v[4]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[3], v[5]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[3], v[6]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[3], v[7]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[3], v[8]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[3], v[9]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[3], v[10]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[3], v[11]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[3], v[12]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[3], v[13]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[4], v[0]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[4], v[1]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[4], v[2]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[4], v[3]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[4], v[4]) == std::strong_ordering::equal);
    assert(std::compare_strong_order_fallback(v[4], v[5]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[4], v[6]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[4], v[7]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[4], v[8]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[4], v[9]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[4], v[10]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[4], v[11]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[4], v[12]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[4], v[13]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[5], v[0]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[5], v[1]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[5], v[2]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[5], v[3]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[5], v[4]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[5], v[5]) == std::strong_ordering::equal);
    assert(std::compare_strong_order_fallback(v[5], v[6]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[5], v[7]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[5], v[8]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[5], v[9]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[5], v[10]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[5], v[11]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[5], v[12]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[5], v[13]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[6], v[0]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[6], v[1]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[6], v[2]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[6], v[3]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[6], v[4]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[6], v[5]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[6], v[6]) == std::strong_ordering::equal);
    assert(std::compare_strong_order_fallback(v[6], v[7]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[6], v[8]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[6], v[9]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[6], v[10]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[6], v[11]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[6], v[12]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[6], v[13]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[7], v[0]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[7], v[1]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[7], v[2]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[7], v[3]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[7], v[4]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[7], v[5]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[7], v[6]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[7], v[7]) == std::strong_ordering::equal);
    assert(std::compare_strong_order_fallback(v[7], v[8]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[7], v[9]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[7], v[10]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[7], v[11]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[7], v[12]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[7], v[13]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[8], v[0]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[8], v[1]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[8], v[2]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[8], v[3]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[8], v[4]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[8], v[5]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[8], v[6]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[8], v[7]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[8], v[8]) == std::strong_ordering::equal);
    assert(std::compare_strong_order_fallback(v[8], v[9]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[8], v[10]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[8], v[11]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[8], v[12]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[8], v[13]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[9], v[0]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[9], v[1]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[9], v[2]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[9], v[3]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[9], v[4]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[9], v[5]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[9], v[6]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[9], v[7]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[9], v[8]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[9], v[9]) == std::strong_ordering::equal);
    assert(std::compare_strong_order_fallback(v[9], v[10]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[9], v[11]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[9], v[12]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[9], v[13]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[10], v[0]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[10], v[1]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[10], v[2]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[10], v[3]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[10], v[4]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[10], v[5]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[10], v[6]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[10], v[7]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[10], v[8]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[10], v[9]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[10], v[10]) == std::strong_ordering::equal);
    assert(std::compare_strong_order_fallback(v[10], v[11]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[10], v[12]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[10], v[13]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[11], v[0]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[11], v[1]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[11], v[2]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[11], v[3]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[11], v[4]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[11], v[5]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[11], v[6]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[11], v[7]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[11], v[8]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[11], v[9]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[11], v[10]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[11], v[11]) == std::strong_ordering::equal);
    assert(std::compare_strong_order_fallback(v[11], v[12]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[11], v[13]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[12], v[0]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[12], v[1]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[12], v[2]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[12], v[3]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[12], v[4]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[12], v[5]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[12], v[6]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[12], v[7]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[12], v[8]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[12], v[9]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[12], v[10]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[12], v[11]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[12], v[12]) == std::strong_ordering::equal);
    assert(std::compare_strong_order_fallback(v[12], v[13]) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(v[13], v[0]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[13], v[1]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[13], v[2]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[13], v[3]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[13], v[4]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[13], v[5]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[13], v[6]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[13], v[7]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[13], v[8]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[13], v[9]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[13], v[10]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[13], v[11]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[13], v[12]) == std::strong_ordering::greater);
    assert(std::compare_strong_order_fallback(v[13], v[13]) == std::strong_ordering::equal);


    // There's no way to produce a specifically positive or negative NAN
    // at compile-time, so the NAN-related tests must be runtime-only.

    if (!std::is_constant_evaluated()) {
        F nq = std::copysign(std::numeric_limits<F>::quiet_NaN(), F(-1));
        F ns = std::copysign(std::numeric_limits<F>::signaling_NaN(), F(-1));
        F ps = std::copysign(std::numeric_limits<F>::signaling_NaN(), F(+1));
        F pq = std::copysign(std::numeric_limits<F>::quiet_NaN(), F(+1));

        assert(std::compare_strong_order_fallback(nq, nq) == std::strong_ordering::equal);
#ifndef TEST_BUGGY_SIGNALING_NAN
        assert(std::compare_strong_order_fallback(nq, ns) == std::strong_ordering::less);
#endif
        for (int i=0; i < 14; ++i) {
            assert(std::compare_strong_order_fallback(nq, v[i]) == std::strong_ordering::less);
        }
        assert(std::compare_strong_order_fallback(nq, ps) == std::strong_ordering::less);
        assert(std::compare_strong_order_fallback(nq, pq) == std::strong_ordering::less);

#ifndef TEST_BUGGY_SIGNALING_NAN
        assert(std::compare_strong_order_fallback(ns, nq) == std::strong_ordering::greater);
#endif
        assert(std::compare_strong_order_fallback(ns, ns) == std::strong_ordering::equal);
        for (int i=0; i < 14; ++i) {
            assert(std::compare_strong_order_fallback(ns, v[i]) == std::strong_ordering::less);
        }
        assert(std::compare_strong_order_fallback(ns, ps) == std::strong_ordering::less);
        assert(std::compare_strong_order_fallback(ns, pq) == std::strong_ordering::less);

        assert(std::compare_strong_order_fallback(ps, nq) == std::strong_ordering::greater);
        assert(std::compare_strong_order_fallback(ps, ns) == std::strong_ordering::greater);
        for (int i=0; i < 14; ++i) {
            assert(std::compare_strong_order_fallback(ps, v[i]) == std::strong_ordering::greater);
        }
        assert(std::compare_strong_order_fallback(ps, ps) == std::strong_ordering::equal);
#ifndef TEST_BUGGY_SIGNALING_NAN
        assert(std::compare_strong_order_fallback(ps, pq) == std::strong_ordering::less);
#endif

        assert(std::compare_strong_order_fallback(pq, nq) == std::strong_ordering::greater);
        assert(std::compare_strong_order_fallback(pq, ns) == std::strong_ordering::greater);
        for (int i=0; i < 14; ++i) {
            assert(std::compare_strong_order_fallback(pq, v[i]) == std::strong_ordering::greater);
        }
#ifndef TEST_BUGGY_SIGNALING_NAN
        assert(std::compare_strong_order_fallback(pq, ps) == std::strong_ordering::greater);
#endif
        assert(std::compare_strong_order_fallback(pq, pq) == std::strong_ordering::equal);
    }

    return true;
}

namespace N14 {
    // Compare to N12::A.
    struct A {};
    bool operator==(const A&, const A&);
    constexpr std::strong_ordering operator<=>(A&, A&&) { return std::strong_ordering::less; }
    constexpr std::strong_ordering operator<=>(A&&, A&&) { return std::strong_ordering::equal; }
    std::strong_ordering operator<=>(const A&, const A&);
    static_assert(std::three_way_comparable<A>);

    struct B {
        std::strong_ordering operator<=>(const B&) const;  // lacks operator==
    };
    static_assert(!std::three_way_comparable<B>);

    struct C {
        bool *touched;
        bool operator==(const C&) const;
        constexpr std::strong_ordering operator<=>(const C& rhs) const {
            *rhs.touched = true;
            return std::strong_ordering::equal;
        }
    };
    static_assert(std::three_way_comparable<C>);
}

constexpr bool test_1_4()
{
    // Otherwise, strong_ordering(compare_three_way()(E, F)) if it is a well-formed expression.

    // Test neither strong_order nor compare_three_way const-qualify the forwarded arguments.
    N14::A a;
    assert(std::compare_strong_order_fallback(a, std::move(a)) == std::strong_ordering::less);
    assert(std::compare_strong_order_fallback(std::move(a), std::move(a)) == std::strong_ordering::equal);

    N14::B b;
    static_assert(!has_strong_order(b, b));

    // Test that the arguments are passed to <=> in the correct order.
    bool c1_touched = false;
    bool c2_touched = false;
    N14::C c1 = {&c1_touched};
    N14::C c2 = {&c2_touched};
    assert(std::compare_strong_order_fallback(c1, c2) == std::strong_ordering::equal);
    assert(!c1_touched);
    assert(c2_touched);

    return true;
}

namespace N2 {
    struct Stats {
        int eq = 0;
        int lt = 0;
    };
    struct A {
        Stats *stats_;
        double value_;
        constexpr explicit A(Stats *stats, double value) : stats_(stats), value_(value) {}
        friend constexpr bool operator==(A a, A b) { a.stats_->eq += 1; return a.value_ == b.value_; }
        friend constexpr bool operator<(A a, A b) { a.stats_->lt += 1; return a.value_ < b.value_; }
    };
    struct NoEquality {
        friend bool operator<(NoEquality, NoEquality);
    };
    struct VC1 {
        // Deliberately asymmetric `const` qualifiers here.
        friend bool operator==(const VC1&, VC1&);
        friend bool operator<(const VC1&, VC1&);
    };
    struct VC2 {
        // Deliberately asymmetric `const` qualifiers here.
        friend bool operator==(const VC2&, VC2&);
        friend bool operator==(VC2&, const VC2&) = delete;
        friend bool operator<(const VC2&, VC2&);
        friend bool operator<(VC2&, const VC2&);
    };
}

constexpr bool test_2()
{
    {
        N2::Stats stats;
        assert(std::compare_strong_order_fallback(N2::A(&stats, 1), N2::A(nullptr, 1)) == std::strong_ordering::equal);
        assert(stats.eq == 1 && stats.lt == 0);
        stats = {};
        assert(std::compare_strong_order_fallback(N2::A(&stats, 1), N2::A(nullptr, 2)) == std::strong_ordering::less);
        assert(stats.eq == 1 && stats.lt == 1);
        stats = {};
        assert(std::compare_strong_order_fallback(N2::A(&stats, 2), N2::A(nullptr, 1)) == std::strong_ordering::greater);
        assert(stats.eq == 1 && stats.lt == 1);
    }
    {
        N2::NoEquality ne;
        assert(!has_strong_order(ne, ne));
    }
    {
        // LWG3465: (cvc < vc) is well-formed, (vc < cvc) is not. That's fine, for strong ordering.
        N2::VC1 vc;
        const N2::VC1 cvc;
        assert( has_strong_order(cvc, vc));
        assert(!has_strong_order(vc, cvc));
    }
    {
        // LWG3465: (cvc == vc) is well-formed, (vc == cvc) is not. That's fine.
        N2::VC2 vc;
        const N2::VC2 cvc;
        assert( has_strong_order(cvc, vc));
        assert(!has_strong_order(vc, cvc));
    }
    return true;
}

int main(int, char**)
{
    test_1_1();
    test_1_2();
    test_1_3<float>();
    test_1_3<double>();
    // test_1_3<long double>();  // UNIMPLEMENTED
    test_1_4();
    test_2();

    static_assert(test_1_3<float>());
    static_assert(test_1_3<double>());
    // static_assert(test_1_3<long double>());  // UNIMPLEMENTED
    static_assert(test_1_4());
    static_assert(test_2());

    return 0;
}
