//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <compare>

// template<class T> constexpr partial_ordering compare_partial_order_fallback(const T& a, const T& b);

#include <compare>

#include <cassert>
#include <cmath>
#include <iterator> // std::size
#include <limits>
#include <type_traits>
#include <utility>

#include "test_macros.h"

template<class T, class U>
constexpr auto has_partial_order(T&& t, U&& u)
    -> decltype(std::compare_partial_order_fallback(static_cast<T&&>(t), static_cast<U&&>(u)), true)
{
    return true;
}

constexpr bool has_partial_order(...) {
    return false;
}

namespace N11 {
    struct A {};
    struct B {};
    std::strong_ordering partial_order(const A&, const A&) { return std::strong_ordering::less; }
    std::strong_ordering partial_order(const A&, const B&);
}

void test_1_1()
{
    // If the decayed types of E and F differ, partial_order(E, F) is ill-formed.

    static_assert( has_partial_order(1, 2));
    static_assert(!has_partial_order(1, (short)2));
    static_assert(!has_partial_order(1, 2.0));
    static_assert(!has_partial_order(1.0f, 2.0));

    static_assert( has_partial_order((int*)nullptr, (int*)nullptr));
    static_assert(!has_partial_order((int*)nullptr, (const int*)nullptr));
    static_assert(!has_partial_order((const int*)nullptr, (int*)nullptr));
    static_assert( has_partial_order((const int*)nullptr, (const int*)nullptr));

    N11::A a;
    N11::B b;
    static_assert( has_partial_order(a, a));
    static_assert(!has_partial_order(a, b));
}

namespace N12 {
    struct A {};
    std::strong_ordering partial_order(A&, A&&) { return std::strong_ordering::less; }
    std::weak_ordering partial_order(A&&, A&&) { return std::weak_ordering::equivalent; }
    std::strong_ordering partial_order(const A&, const A&);

    struct B {
        friend int partial_order(B, B);
    };

    struct PartialOrder {
        explicit operator std::partial_ordering() const { return std::partial_ordering::less; }
    };
    struct C {
        bool touched = false;
        friend PartialOrder partial_order(C& lhs, C&) { lhs.touched = true; return PartialOrder(); }
    };
}

void test_1_2()
{
    // Otherwise, partial_ordering(partial_order(E, F))
    // if it is a well-formed expression with overload resolution performed
    // in a context that does not include a declaration of std::partial_order.

    // Test that partial_order does not const-qualify the forwarded arguments.
    N12::A a;
    assert(std::compare_partial_order_fallback(a, std::move(a)) == std::partial_ordering::less);
    assert(std::compare_partial_order_fallback(std::move(a), std::move(a)) == std::partial_ordering::equivalent);

    // The type of partial_order(e,f) must be explicitly convertible to partial_ordering.
    N12::B b;
    static_assert(!has_partial_order(b, b));

    N12::C c1, c2;
    ASSERT_SAME_TYPE(decltype(std::compare_partial_order_fallback(c1, c2)), std::partial_ordering);
    assert(std::partial_order(c1, c2) == std::partial_ordering::less);
    assert(c1.touched);
    assert(!c2.touched);
}

namespace N13 {
    // Compare to N12::A.
    struct A {};
    bool operator==(const A&, const A&);
    constexpr std::partial_ordering operator<=>(A&, A&&) { return std::partial_ordering::less; }
    constexpr std::partial_ordering operator<=>(A&&, A&&) { return std::partial_ordering::equivalent; }
    std::partial_ordering operator<=>(const A&, const A&);
    static_assert(std::three_way_comparable<A>);

    struct B {
        std::partial_ordering operator<=>(const B&) const;  // lacks operator==
    };
    static_assert(!std::three_way_comparable<B>);

    struct C {
        bool *touched;
        bool operator==(const C&) const;
        constexpr std::partial_ordering operator<=>(const C& rhs) const {
            *rhs.touched = true;
            return std::partial_ordering::equivalent;
        }
    };
    static_assert(std::three_way_comparable<C>);
}

constexpr bool test_1_3()
{
    // Otherwise, partial_ordering(compare_three_way()(E, F)) if it is a well-formed expression.

    // Test neither partial_order nor compare_three_way const-qualify the forwarded arguments.
    N13::A a;
    assert(std::compare_partial_order_fallback(a, std::move(a)) == std::partial_ordering::less);
    assert(std::compare_partial_order_fallback(std::move(a), std::move(a)) == std::partial_ordering::equivalent);

    N13::B b;
    static_assert(!has_partial_order(b, b));

    // Test that the arguments are passed to <=> in the correct order.
    bool c1_touched = false;
    bool c2_touched = false;
    N13::C c1 = {&c1_touched};
    N13::C c2 = {&c2_touched};
    assert(std::compare_partial_order_fallback(c1, c2) == std::partial_ordering::equivalent);
    assert(!c1_touched);
    assert(c2_touched);

    // For partial_order, this bullet point takes care of floating-point types;
    // they receive their natural partial order.
    {
        using F = float;
        F nan = std::numeric_limits<F>::quiet_NaN();
        assert(std::compare_partial_order_fallback(F(1), F(2)) == std::partial_ordering::less);
        assert(std::compare_partial_order_fallback(F(0), -F(0)) == std::partial_ordering::equivalent);
#ifndef TEST_COMPILER_GCC  // GCC can't compare NaN to non-NaN in a constant-expression
        assert(std::compare_partial_order_fallback(nan, F(1)) == std::partial_ordering::unordered);
#endif
        assert(std::compare_partial_order_fallback(nan, nan) == std::partial_ordering::unordered);
    }
    {
        using F = double;
        F nan = std::numeric_limits<F>::quiet_NaN();
        assert(std::compare_partial_order_fallback(F(1), F(2)) == std::partial_ordering::less);
        assert(std::compare_partial_order_fallback(F(0), -F(0)) == std::partial_ordering::equivalent);
#ifndef TEST_COMPILER_GCC
        assert(std::compare_partial_order_fallback(nan, F(1)) == std::partial_ordering::unordered);
#endif
        assert(std::compare_partial_order_fallback(nan, nan) == std::partial_ordering::unordered);
    }
    {
        using F = long double;
        F nan = std::numeric_limits<F>::quiet_NaN();
        assert(std::compare_partial_order_fallback(F(1), F(2)) == std::partial_ordering::less);
        assert(std::compare_partial_order_fallback(F(0), -F(0)) == std::partial_ordering::equivalent);
#ifndef TEST_COMPILER_GCC
        assert(std::compare_partial_order_fallback(nan, F(1)) == std::partial_ordering::unordered);
#endif
        assert(std::compare_partial_order_fallback(nan, nan) == std::partial_ordering::unordered);
    }

    return true;
}

namespace N14 {
    struct A {};
    constexpr std::strong_ordering weak_order(A&, A&&) { return std::strong_ordering::less; }
    constexpr std::strong_ordering weak_order(A&&, A&&) { return std::strong_ordering::equal; }
    std::strong_ordering weak_order(const A&, const A&);

    struct B {
        friend std::partial_ordering weak_order(B, B);
    };

    struct StrongOrder {
        operator std::strong_ordering() const { return std::strong_ordering::less; }
    };
    struct C {
        friend StrongOrder weak_order(C& lhs, C&);
    };

    struct WeakOrder {
        constexpr explicit operator std::weak_ordering() const { return std::weak_ordering::less; }
        operator std::partial_ordering() const = delete;
    };
    struct D {
        bool touched = false;
        friend constexpr WeakOrder weak_order(D& lhs, D&) { lhs.touched = true; return WeakOrder(); }
    };
}

constexpr bool test_1_4()
{
    // Otherwise, partial_ordering(weak_order(E, F)) [that is, std::weak_order]
    // if it is a well-formed expression.

    // Test that partial_order and weak_order do not const-qualify the forwarded arguments.
    N14::A a;
    assert(std::compare_partial_order_fallback(a, std::move(a)) == std::partial_ordering::less);
    assert(std::compare_partial_order_fallback(std::move(a), std::move(a)) == std::partial_ordering::equivalent);

    // The type of ADL weak_order(e,f) must be explicitly convertible to weak_ordering
    // (not just to partial_ordering), or else std::weak_order(e,f) won't exist.
    N14::B b;
    static_assert(!has_partial_order(b, b));

    // The type of ADL weak_order(e,f) must be explicitly convertible to weak_ordering
    // (not just to strong_ordering), or else std::weak_order(e,f) won't exist.
    N14::C c;
    static_assert(!has_partial_order(c, c));

    N14::D d1, d2;
    ASSERT_SAME_TYPE(decltype(std::compare_partial_order_fallback(d1, d2)), std::partial_ordering);
    assert(std::compare_partial_order_fallback(d1, d2) == std::partial_ordering::less);
    assert(d1.touched);
    assert(!d2.touched);

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
        N2::Stats bstats;
        assert(std::compare_partial_order_fallback(N2::A(&stats, 1), N2::A(nullptr, 1)) == std::partial_ordering::equivalent);
        assert(stats.eq == 1 && stats.lt == 0);
        stats = {};
        assert(std::compare_partial_order_fallback(N2::A(&stats, 1), N2::A(nullptr, 2)) == std::partial_ordering::less);
        assert(stats.eq == 1 && stats.lt == 1);
        stats = {};
        assert(std::compare_partial_order_fallback(N2::A(&stats, 2), N2::A(&bstats, 1)) == std::partial_ordering::greater);
        assert(stats.eq == 1 && stats.lt == 1 && bstats.lt == 1);
        stats = {};
        bstats = {};
        double nan = std::numeric_limits<double>::quiet_NaN();
        assert(std::compare_partial_order_fallback(N2::A(&stats, nan), N2::A(&bstats, nan)) == std::partial_ordering::unordered);
        assert(stats.eq == 1 && stats.lt == 1 && bstats.lt == 1);
    }
    {
        N2::NoEquality ne;
        assert(!has_partial_order(ne, ne));
    }
    {
        // LWG3465: (cvc < vc) is well-formed, (vc < cvc) is not. Substitution failure.
        N2::VC1 vc;
        const N2::VC1 cvc;
        assert(!has_partial_order(cvc, vc));
        assert(!has_partial_order(vc, cvc));
    }
    {
        // LWG3465: (cvc == vc) is well-formed, (vc == cvc) is not. That's fine.
        N2::VC2 vc;
        const N2::VC2 cvc;
        assert( has_partial_order(cvc, vc));
        assert(!has_partial_order(vc, cvc));
    }
    return true;
}

int main(int, char**)
{
    test_1_1();
    test_1_2();
    test_1_3();
    test_1_4();
    test_2();

    static_assert(test_1_3());
    static_assert(test_1_4());
    static_assert(test_2());

    return 0;
}
