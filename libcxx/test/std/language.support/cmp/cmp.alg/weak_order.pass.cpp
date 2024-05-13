//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <compare>

// template<class T> constexpr weak_ordering weak_order(const T& a, const T& b);

#include <compare>

#include <cassert>
#include <cmath>
#include <iterator> // std::size
#include <limits>
#include <type_traits>
#include <utility>

#include "test_macros.h"

template<class T, class U>
constexpr auto has_weak_order(T&& t, U&& u)
    -> decltype(std::weak_order(static_cast<T&&>(t), static_cast<U&&>(u)), true)
{
    return true;
}

constexpr bool has_weak_order(...) {
    return false;
}

namespace N11 {
    struct A {};
    struct B {};
    std::strong_ordering weak_order(const A&, const A&) { return std::strong_ordering::less; }
    std::strong_ordering weak_order(const A&, const B&);
}

void test_1_1()
{
    // If the decayed types of E and F differ, weak_order(E, F) is ill-formed.

    static_assert( has_weak_order(1, 2));
    static_assert(!has_weak_order(1, (short)2));
    static_assert(!has_weak_order(1, 2.0));
    static_assert(!has_weak_order(1.0f, 2.0));

    static_assert( has_weak_order((int*)nullptr, (int*)nullptr));
    static_assert(!has_weak_order((int*)nullptr, (const int*)nullptr));
    static_assert(!has_weak_order((const int*)nullptr, (int*)nullptr));
    static_assert( has_weak_order((const int*)nullptr, (const int*)nullptr));

    N11::A a;
    N11::B b;
    static_assert( has_weak_order(a, a));
    static_assert(!has_weak_order(a, b));
}

namespace N12 {
    struct A {};
    std::strong_ordering weak_order(A&, A&&) { return std::strong_ordering::less; }
    std::strong_ordering weak_order(A&&, A&&) { return std::strong_ordering::equal; }
    std::strong_ordering weak_order(const A&, const A&);

    struct B {
        friend std::partial_ordering weak_order(B&, B&);
    };

    struct WeakOrder {
        explicit operator std::weak_ordering() const { return std::weak_ordering::less; }
    };
    struct C {
        bool touched = false;
        friend WeakOrder weak_order(C& lhs, C&) { lhs.touched = true; return WeakOrder(); }
    };
}

void test_1_2()
{
    // Otherwise, weak_ordering(weak_order(E, F))
    // if it is a well-formed expression with overload resolution performed
    // in a context that does not include a declaration of std::weak_order.

    // Test that weak_order does not const-qualify the forwarded arguments.
    N12::A a;
    assert(std::weak_order(a, std::move(a)) == std::weak_ordering::less);
    assert(std::weak_order(std::move(a), std::move(a)) == std::weak_ordering::equivalent);

    // The type of weak_order(e,f) must be explicitly convertible to weak_ordering.
    N12::B b;
    static_assert(!has_weak_order(b, b));

    N12::C c1, c2;
    ASSERT_SAME_TYPE(decltype(std::weak_order(c1, c2)), std::weak_ordering);
    assert(std::weak_order(c1, c2) == std::weak_ordering::less);
    assert(c1.touched);
    assert(!c2.touched);
}

template<class F>
constexpr bool test_1_3()
{
    // Otherwise, if the decayed type T of E is a floating-point type,
    // yields a value of type weak_ordering that is consistent with
    // the ordering observed by T's comparison operators and strong_order,
    // and if numeric_limits<T>::is_iec559 is true, is additionally consistent with
    // the following equivalence classes...

    // std::numeric_limits<F>::is_iec559 is usually true.
    // It is false for F=long double on AIX; but this test is still expected
    // to pass (e.g. std::weak_order(+0, -0) == weak_ordering::equivalent,
    // even on AIX).

    ASSERT_SAME_TYPE(decltype(std::weak_order(F(0), F(0))), std::weak_ordering);

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

    assert(std::weak_order(v[0], v[0]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[0], v[1]) == std::weak_ordering::less);
    assert(std::weak_order(v[0], v[2]) == std::weak_ordering::less);
    assert(std::weak_order(v[0], v[3]) == std::weak_ordering::less);
    assert(std::weak_order(v[0], v[4]) == std::weak_ordering::less);
    assert(std::weak_order(v[0], v[5]) == std::weak_ordering::less);
    assert(std::weak_order(v[0], v[6]) == std::weak_ordering::less);
    assert(std::weak_order(v[0], v[7]) == std::weak_ordering::less);
    assert(std::weak_order(v[0], v[8]) == std::weak_ordering::less);
    assert(std::weak_order(v[0], v[9]) == std::weak_ordering::less);
    assert(std::weak_order(v[0], v[10]) == std::weak_ordering::less);
    assert(std::weak_order(v[0], v[11]) == std::weak_ordering::less);
    assert(std::weak_order(v[0], v[12]) == std::weak_ordering::less);
    assert(std::weak_order(v[0], v[13]) == std::weak_ordering::less);
    assert(std::weak_order(v[1], v[0]) == std::weak_ordering::greater);
    assert(std::weak_order(v[1], v[1]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[1], v[2]) == std::weak_ordering::less);
    assert(std::weak_order(v[1], v[3]) == std::weak_ordering::less);
    assert(std::weak_order(v[1], v[4]) == std::weak_ordering::less);
    assert(std::weak_order(v[1], v[5]) == std::weak_ordering::less);
    assert(std::weak_order(v[1], v[6]) == std::weak_ordering::less);
    assert(std::weak_order(v[1], v[7]) == std::weak_ordering::less);
    assert(std::weak_order(v[1], v[8]) == std::weak_ordering::less);
    assert(std::weak_order(v[1], v[9]) == std::weak_ordering::less);
    assert(std::weak_order(v[1], v[10]) == std::weak_ordering::less);
    assert(std::weak_order(v[1], v[11]) == std::weak_ordering::less);
    assert(std::weak_order(v[1], v[12]) == std::weak_ordering::less);
    assert(std::weak_order(v[1], v[13]) == std::weak_ordering::less);
    assert(std::weak_order(v[2], v[0]) == std::weak_ordering::greater);
    assert(std::weak_order(v[2], v[1]) == std::weak_ordering::greater);
    assert(std::weak_order(v[2], v[2]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[2], v[3]) == std::weak_ordering::less);
    assert(std::weak_order(v[2], v[4]) == std::weak_ordering::less);
    assert(std::weak_order(v[2], v[5]) == std::weak_ordering::less);
    assert(std::weak_order(v[2], v[6]) == std::weak_ordering::less);
    assert(std::weak_order(v[2], v[7]) == std::weak_ordering::less);
    assert(std::weak_order(v[2], v[8]) == std::weak_ordering::less);
    assert(std::weak_order(v[2], v[9]) == std::weak_ordering::less);
    assert(std::weak_order(v[2], v[10]) == std::weak_ordering::less);
    assert(std::weak_order(v[2], v[11]) == std::weak_ordering::less);
    assert(std::weak_order(v[2], v[12]) == std::weak_ordering::less);
    assert(std::weak_order(v[2], v[13]) == std::weak_ordering::less);
    assert(std::weak_order(v[3], v[0]) == std::weak_ordering::greater);
    assert(std::weak_order(v[3], v[1]) == std::weak_ordering::greater);
    assert(std::weak_order(v[3], v[2]) == std::weak_ordering::greater);
    assert(std::weak_order(v[3], v[3]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[3], v[4]) == std::weak_ordering::less);
    assert(std::weak_order(v[3], v[5]) == std::weak_ordering::less);
    assert(std::weak_order(v[3], v[6]) == std::weak_ordering::less);
    assert(std::weak_order(v[3], v[7]) == std::weak_ordering::less);
    assert(std::weak_order(v[3], v[8]) == std::weak_ordering::less);
    assert(std::weak_order(v[3], v[9]) == std::weak_ordering::less);
    assert(std::weak_order(v[3], v[10]) == std::weak_ordering::less);
    assert(std::weak_order(v[3], v[11]) == std::weak_ordering::less);
    assert(std::weak_order(v[3], v[12]) == std::weak_ordering::less);
    assert(std::weak_order(v[3], v[13]) == std::weak_ordering::less);
    assert(std::weak_order(v[4], v[0]) == std::weak_ordering::greater);
    assert(std::weak_order(v[4], v[1]) == std::weak_ordering::greater);
    assert(std::weak_order(v[4], v[2]) == std::weak_ordering::greater);
    assert(std::weak_order(v[4], v[3]) == std::weak_ordering::greater);
    assert(std::weak_order(v[4], v[4]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[4], v[5]) == std::weak_ordering::less);
    assert(std::weak_order(v[4], v[6]) == std::weak_ordering::less);
    assert(std::weak_order(v[4], v[7]) == std::weak_ordering::less);
    assert(std::weak_order(v[4], v[8]) == std::weak_ordering::less);
    assert(std::weak_order(v[4], v[9]) == std::weak_ordering::less);
    assert(std::weak_order(v[4], v[10]) == std::weak_ordering::less);
    assert(std::weak_order(v[4], v[11]) == std::weak_ordering::less);
    assert(std::weak_order(v[4], v[12]) == std::weak_ordering::less);
    assert(std::weak_order(v[4], v[13]) == std::weak_ordering::less);
    assert(std::weak_order(v[5], v[0]) == std::weak_ordering::greater);
    assert(std::weak_order(v[5], v[1]) == std::weak_ordering::greater);
    assert(std::weak_order(v[5], v[2]) == std::weak_ordering::greater);
    assert(std::weak_order(v[5], v[3]) == std::weak_ordering::greater);
    assert(std::weak_order(v[5], v[4]) == std::weak_ordering::greater);
    assert(std::weak_order(v[5], v[5]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[5], v[6]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[5], v[7]) == std::weak_ordering::less);
    assert(std::weak_order(v[5], v[8]) == std::weak_ordering::less);
    assert(std::weak_order(v[5], v[9]) == std::weak_ordering::less);
    assert(std::weak_order(v[5], v[10]) == std::weak_ordering::less);
    assert(std::weak_order(v[5], v[11]) == std::weak_ordering::less);
    assert(std::weak_order(v[5], v[12]) == std::weak_ordering::less);
    assert(std::weak_order(v[5], v[13]) == std::weak_ordering::less);
    assert(std::weak_order(v[6], v[0]) == std::weak_ordering::greater);
    assert(std::weak_order(v[6], v[1]) == std::weak_ordering::greater);
    assert(std::weak_order(v[6], v[2]) == std::weak_ordering::greater);
    assert(std::weak_order(v[6], v[3]) == std::weak_ordering::greater);
    assert(std::weak_order(v[6], v[4]) == std::weak_ordering::greater);
    assert(std::weak_order(v[6], v[5]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[6], v[6]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[6], v[7]) == std::weak_ordering::less);
    assert(std::weak_order(v[6], v[8]) == std::weak_ordering::less);
    assert(std::weak_order(v[6], v[9]) == std::weak_ordering::less);
    assert(std::weak_order(v[6], v[10]) == std::weak_ordering::less);
    assert(std::weak_order(v[6], v[11]) == std::weak_ordering::less);
    assert(std::weak_order(v[6], v[12]) == std::weak_ordering::less);
    assert(std::weak_order(v[6], v[13]) == std::weak_ordering::less);
    assert(std::weak_order(v[7], v[0]) == std::weak_ordering::greater);
    assert(std::weak_order(v[7], v[1]) == std::weak_ordering::greater);
    assert(std::weak_order(v[7], v[2]) == std::weak_ordering::greater);
    assert(std::weak_order(v[7], v[3]) == std::weak_ordering::greater);
    assert(std::weak_order(v[7], v[4]) == std::weak_ordering::greater);
    assert(std::weak_order(v[7], v[5]) == std::weak_ordering::greater);
    assert(std::weak_order(v[7], v[6]) == std::weak_ordering::greater);
    assert(std::weak_order(v[7], v[7]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[7], v[8]) == std::weak_ordering::less);
    assert(std::weak_order(v[7], v[9]) == std::weak_ordering::less);
    assert(std::weak_order(v[7], v[10]) == std::weak_ordering::less);
    assert(std::weak_order(v[7], v[11]) == std::weak_ordering::less);
    assert(std::weak_order(v[7], v[12]) == std::weak_ordering::less);
    assert(std::weak_order(v[7], v[13]) == std::weak_ordering::less);
    assert(std::weak_order(v[8], v[0]) == std::weak_ordering::greater);
    assert(std::weak_order(v[8], v[1]) == std::weak_ordering::greater);
    assert(std::weak_order(v[8], v[2]) == std::weak_ordering::greater);
    assert(std::weak_order(v[8], v[3]) == std::weak_ordering::greater);
    assert(std::weak_order(v[8], v[4]) == std::weak_ordering::greater);
    assert(std::weak_order(v[8], v[5]) == std::weak_ordering::greater);
    assert(std::weak_order(v[8], v[6]) == std::weak_ordering::greater);
    assert(std::weak_order(v[8], v[7]) == std::weak_ordering::greater);
    assert(std::weak_order(v[8], v[8]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[8], v[9]) == std::weak_ordering::less);
    assert(std::weak_order(v[8], v[10]) == std::weak_ordering::less);
    assert(std::weak_order(v[8], v[11]) == std::weak_ordering::less);
    assert(std::weak_order(v[8], v[12]) == std::weak_ordering::less);
    assert(std::weak_order(v[8], v[13]) == std::weak_ordering::less);
    assert(std::weak_order(v[9], v[0]) == std::weak_ordering::greater);
    assert(std::weak_order(v[9], v[1]) == std::weak_ordering::greater);
    assert(std::weak_order(v[9], v[2]) == std::weak_ordering::greater);
    assert(std::weak_order(v[9], v[3]) == std::weak_ordering::greater);
    assert(std::weak_order(v[9], v[4]) == std::weak_ordering::greater);
    assert(std::weak_order(v[9], v[5]) == std::weak_ordering::greater);
    assert(std::weak_order(v[9], v[6]) == std::weak_ordering::greater);
    assert(std::weak_order(v[9], v[7]) == std::weak_ordering::greater);
    assert(std::weak_order(v[9], v[8]) == std::weak_ordering::greater);
    assert(std::weak_order(v[9], v[9]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[9], v[10]) == std::weak_ordering::less);
    assert(std::weak_order(v[9], v[11]) == std::weak_ordering::less);
    assert(std::weak_order(v[9], v[12]) == std::weak_ordering::less);
    assert(std::weak_order(v[9], v[13]) == std::weak_ordering::less);
    assert(std::weak_order(v[10], v[0]) == std::weak_ordering::greater);
    assert(std::weak_order(v[10], v[1]) == std::weak_ordering::greater);
    assert(std::weak_order(v[10], v[2]) == std::weak_ordering::greater);
    assert(std::weak_order(v[10], v[3]) == std::weak_ordering::greater);
    assert(std::weak_order(v[10], v[4]) == std::weak_ordering::greater);
    assert(std::weak_order(v[10], v[5]) == std::weak_ordering::greater);
    assert(std::weak_order(v[10], v[6]) == std::weak_ordering::greater);
    assert(std::weak_order(v[10], v[7]) == std::weak_ordering::greater);
    assert(std::weak_order(v[10], v[8]) == std::weak_ordering::greater);
    assert(std::weak_order(v[10], v[9]) == std::weak_ordering::greater);
    assert(std::weak_order(v[10], v[10]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[10], v[11]) == std::weak_ordering::less);
    assert(std::weak_order(v[10], v[12]) == std::weak_ordering::less);
    assert(std::weak_order(v[10], v[13]) == std::weak_ordering::less);
    assert(std::weak_order(v[11], v[0]) == std::weak_ordering::greater);
    assert(std::weak_order(v[11], v[1]) == std::weak_ordering::greater);
    assert(std::weak_order(v[11], v[2]) == std::weak_ordering::greater);
    assert(std::weak_order(v[11], v[3]) == std::weak_ordering::greater);
    assert(std::weak_order(v[11], v[4]) == std::weak_ordering::greater);
    assert(std::weak_order(v[11], v[5]) == std::weak_ordering::greater);
    assert(std::weak_order(v[11], v[6]) == std::weak_ordering::greater);
    assert(std::weak_order(v[11], v[7]) == std::weak_ordering::greater);
    assert(std::weak_order(v[11], v[8]) == std::weak_ordering::greater);
    assert(std::weak_order(v[11], v[9]) == std::weak_ordering::greater);
    assert(std::weak_order(v[11], v[10]) == std::weak_ordering::greater);
    assert(std::weak_order(v[11], v[11]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[11], v[12]) == std::weak_ordering::less);
    assert(std::weak_order(v[11], v[13]) == std::weak_ordering::less);
    assert(std::weak_order(v[12], v[0]) == std::weak_ordering::greater);
    assert(std::weak_order(v[12], v[1]) == std::weak_ordering::greater);
    assert(std::weak_order(v[12], v[2]) == std::weak_ordering::greater);
    assert(std::weak_order(v[12], v[3]) == std::weak_ordering::greater);
    assert(std::weak_order(v[12], v[4]) == std::weak_ordering::greater);
    assert(std::weak_order(v[12], v[5]) == std::weak_ordering::greater);
    assert(std::weak_order(v[12], v[6]) == std::weak_ordering::greater);
    assert(std::weak_order(v[12], v[7]) == std::weak_ordering::greater);
    assert(std::weak_order(v[12], v[8]) == std::weak_ordering::greater);
    assert(std::weak_order(v[12], v[9]) == std::weak_ordering::greater);
    assert(std::weak_order(v[12], v[10]) == std::weak_ordering::greater);
    assert(std::weak_order(v[12], v[11]) == std::weak_ordering::greater);
    assert(std::weak_order(v[12], v[12]) == std::weak_ordering::equivalent);
    assert(std::weak_order(v[12], v[13]) == std::weak_ordering::less);
    assert(std::weak_order(v[13], v[0]) == std::weak_ordering::greater);
    assert(std::weak_order(v[13], v[1]) == std::weak_ordering::greater);
    assert(std::weak_order(v[13], v[2]) == std::weak_ordering::greater);
    assert(std::weak_order(v[13], v[3]) == std::weak_ordering::greater);
    assert(std::weak_order(v[13], v[4]) == std::weak_ordering::greater);
    assert(std::weak_order(v[13], v[5]) == std::weak_ordering::greater);
    assert(std::weak_order(v[13], v[6]) == std::weak_ordering::greater);
    assert(std::weak_order(v[13], v[7]) == std::weak_ordering::greater);
    assert(std::weak_order(v[13], v[8]) == std::weak_ordering::greater);
    assert(std::weak_order(v[13], v[9]) == std::weak_ordering::greater);
    assert(std::weak_order(v[13], v[10]) == std::weak_ordering::greater);
    assert(std::weak_order(v[13], v[11]) == std::weak_ordering::greater);
    assert(std::weak_order(v[13], v[12]) == std::weak_ordering::greater);
    assert(std::weak_order(v[13], v[13]) == std::weak_ordering::equivalent);


    // There's no way to produce a specifically positive or negative NAN
    // at compile-time, so the NAN-related tests must be runtime-only.

    if (!std::is_constant_evaluated()) {
        F nq = std::copysign(std::numeric_limits<F>::quiet_NaN(), F(-1));
        F ns = std::copysign(std::numeric_limits<F>::signaling_NaN(), F(-1));
        F ps = std::copysign(std::numeric_limits<F>::signaling_NaN(), F(+1));
        F pq = std::copysign(std::numeric_limits<F>::quiet_NaN(), F(+1));

        assert(std::weak_order(nq, nq) == std::weak_ordering::equivalent);
        assert(std::weak_order(nq, ns) == std::weak_ordering::equivalent);
        for (int i=0; i < 14; ++i) {
            assert(std::weak_order(nq, v[i]) == std::weak_ordering::less);
        }
        assert(std::weak_order(nq, ps) == std::weak_ordering::less);
        assert(std::weak_order(nq, pq) == std::weak_ordering::less);

        assert(std::weak_order(ns, nq) == std::weak_ordering::equivalent);
        assert(std::weak_order(ns, ns) == std::weak_ordering::equivalent);
        for (int i=0; i < 14; ++i) {
            assert(std::weak_order(ns, v[i]) == std::weak_ordering::less);
        }
        assert(std::weak_order(ns, ps) == std::weak_ordering::less);
        assert(std::weak_order(ns, pq) == std::weak_ordering::less);

        assert(std::weak_order(ps, nq) == std::weak_ordering::greater);
        assert(std::weak_order(ps, ns) == std::weak_ordering::greater);
        for (int i=0; i < 14; ++i) {
            assert(std::weak_order(ps, v[i]) == std::weak_ordering::greater);
        }
        assert(std::weak_order(ps, ps) == std::weak_ordering::equivalent);
        assert(std::weak_order(ps, pq) == std::weak_ordering::equivalent);

        assert(std::weak_order(pq, nq) == std::weak_ordering::greater);
        assert(std::weak_order(pq, ns) == std::weak_ordering::greater);
        for (int i=0; i < 14; ++i) {
            assert(std::weak_order(pq, v[i]) == std::weak_ordering::greater);
        }
        assert(std::weak_order(pq, ps) == std::weak_ordering::equivalent);
        assert(std::weak_order(pq, pq) == std::weak_ordering::equivalent);
    }

    return true;
}

namespace N14 {
    // Compare to N12::A.
    struct A {};
    bool operator==(const A&, const A&);
    constexpr std::weak_ordering operator<=>(A&, A&&) { return std::weak_ordering::less; }
    constexpr std::weak_ordering operator<=>(A&&, A&&) { return std::weak_ordering::equivalent; }
    std::weak_ordering operator<=>(const A&, const A&);
    static_assert(std::three_way_comparable<A>);

    struct B {
        std::weak_ordering operator<=>(const B&) const;  // lacks operator==
    };
    static_assert(!std::three_way_comparable<B>);

    struct C {
        bool *touched;
        bool operator==(const C&) const;
        constexpr std::weak_ordering operator<=>(const C& rhs) const {
            *rhs.touched = true;
            return std::weak_ordering::equivalent;
        }
    };
    static_assert(std::three_way_comparable<C>);
}

constexpr bool test_1_4()
{
    // Otherwise, weak_ordering(compare_three_way()(E, F)) if it is a well-formed expression.

    // Test neither weak_order nor compare_three_way const-qualify the forwarded arguments.
    N14::A a;
    assert(std::weak_order(a, std::move(a)) == std::weak_ordering::less);
    assert(std::weak_order(std::move(a), std::move(a)) == std::weak_ordering::equivalent);

    N14::B b;
    static_assert(!has_weak_order(b, b));

    // Test that the arguments are passed to <=> in the correct order.
    bool c1_touched = false;
    bool c2_touched = false;
    N14::C c1 = {&c1_touched};
    N14::C c2 = {&c2_touched};
    assert(std::weak_order(c1, c2) == std::weak_ordering::equivalent);
    assert(!c1_touched);
    assert(c2_touched);

    return true;
}

namespace N15 {
    struct A {};
    constexpr std::strong_ordering strong_order(A&, A&&) { return std::strong_ordering::less; }
    constexpr std::strong_ordering strong_order(A&&, A&&) { return std::strong_ordering::equal; }
    std::strong_ordering strong_order(const A&, const A&);

    struct B {
        friend std::weak_ordering strong_order(B&, B&);
    };

    struct WeakOrder {
        operator std::weak_ordering() const { return std::weak_ordering::less; }
    };
    struct C {
        friend WeakOrder strong_order(C& lhs, C&);
    };

    struct StrongOrder {
        constexpr explicit operator std::strong_ordering() const { return std::strong_ordering::less; }
        operator std::weak_ordering() const = delete;
    };
    struct D {
        bool touched = false;
        friend constexpr StrongOrder strong_order(D& lhs, D&) { lhs.touched = true; return StrongOrder(); }
    };
}

constexpr bool test_1_5()
{
    // Otherwise, weak_ordering(strong_order(E, F)) [that is, std::strong_order]
    // if it is a well-formed expression.

    // Test that weak_order and strong_order do not const-qualify the forwarded arguments.
    N15::A a;
    assert(std::weak_order(a, std::move(a)) == std::weak_ordering::less);
    assert(std::weak_order(std::move(a), std::move(a)) == std::weak_ordering::equivalent);

    // The type of ADL strong_order(e,f) must be explicitly convertible to strong_ordering
    // (not just to weak_ordering), or else std::strong_order(e,f) won't exist.
    N15::B b;
    static_assert(!has_weak_order(b, b));

    // The type of ADL strong_order(e,f) must be explicitly convertible to strong_ordering
    // (not just to weak_ordering), or else std::strong_order(e,f) won't exist.
    N15::C c;
    static_assert(!has_weak_order(c, c));

    N15::D d1, d2;
    ASSERT_SAME_TYPE(decltype(std::weak_order(d1, d2)), std::weak_ordering);
    assert(std::weak_order(d1, d2) == std::weak_ordering::less);
    assert(d1.touched);
    assert(!d2.touched);

    return true;
}

int main(int, char**)
{
    test_1_1();
    test_1_2();
    test_1_3<float>();
    test_1_3<double>();
    test_1_3<long double>();
    test_1_4();
    test_1_5();

    static_assert(test_1_3<float>());
    static_assert(test_1_3<double>());
    static_assert(test_1_3<long double>());
    static_assert(test_1_4());
    static_assert(test_1_5());

    return 0;
}
