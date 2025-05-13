//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template<class... TTypes, class... UTypes>
// constexpr bool operator==(const tuple<TTypes...>&, const tuple<UTypes...>&); // constexpr since C++14

// constexpr bool operator==(const tuple<TTypes...>&, const UTuple&);
// template<class... TTypes, class... UTypes>                                   // Since C++23

// UNSUPPORTED: c++03

#include <array>
#include <cassert>
#include <tuple>

#include "test_comparisons.h"
#include "test_macros.h"

#if TEST_STD_VER >= 26

// Test SFINAE.

// ==(const tuple<>&, const tuple<>&);

static_assert(std::equality_comparable<std::tuple<EqualityComparable>>);
static_assert(std::equality_comparable<std::tuple<EqualityComparable, EqualityComparable>>);

static_assert(!std::equality_comparable<std::tuple<NonComparable>>);
static_assert(!std::equality_comparable<std::tuple<EqualityComparable, NonComparable>>);
static_assert(!std::equality_comparable<std::tuple<NonComparable, EqualityComparable>>);
static_assert(!std::equality_comparable_with<std::tuple<EqualityComparable>, std::tuple<NonComparable>>);
static_assert(!std::equality_comparable_with<std::tuple<NonComparable>, std::tuple<EqualityComparable>>);
// Size mismatch.
static_assert(!std::equality_comparable_with<std::tuple<EqualityComparable>, std::tuple<EqualityComparable, EqualityComparable>>);
static_assert(!std::equality_comparable_with<std::tuple<EqualityComparable, EqualityComparable>, std::tuple<EqualityComparable>>);

// ==(const tuple<>&, const tuple-like&);

// static_assert(std::equality_comparable_with<std::tuple<EqualityComparable, EqualityComparable>, std::pair<EqualityComparable, EqualityComparable>>);
// static_assert(std::equality_comparable_with<std::tuple<EqualityComparable, EqualityComparable>, std::array<EqualityComparable, 2>>);

// static_assert(!std::equality_comparable_with<std::tuple<EqualityComparable, NonComparable>, std::pair<EqualityComparable, NonComparable>>);
// static_assert(!std::equality_comparable_with<std::tuple<EqualityComparable, NonComparable>, std::array<EqualityComparable, 2>>);
// // Size mismatch.
// static_assert(!std::equality_comparable_with<std::tuple<EqualityComparable>, std::pair<EqualityComparable, EqualityComparable>>);
// static_assert(!std::equality_comparable_with<std::tuple<EqualityComparable, EqualityComparable>, std::array<EqualityComparable, 94>>);

#endif

int main(int, char**)
{
    // {
    //     using T1 = std::tuple<int>;
    //     using T2 = std::array<int, 1>;
    //     const T1 t1(1);
    //     const T2 t2{1};
    //     assert(t1 == t2);
    //     assert(!(t1 != t2));
    // }
    {
        typedef std::tuple<> T1;
        typedef std::tuple<> T2;
        const T1 t1;
        const T2 t2;
        assert(t1 == t2);
        assert(!(t1 != t2));
    }
    {
        typedef std::tuple<int> T1;
        typedef std::tuple<double> T2;
        const T1 t1(1);
        const T2 t2(1.1);
        assert(!(t1 == t2));
        assert(t1 != t2);
    }
    {
        typedef std::tuple<int> T1;
        typedef std::tuple<double> T2;
        const T1 t1(1);
        const T2 t2(1);
        assert(t1 == t2);
        assert(!(t1 != t2));
    }
    {
        typedef std::tuple<int, double> T1;
        typedef std::tuple<double, long> T2;
        const T1 t1(1, 2);
        const T2 t2(1, 2);
        assert(t1 == t2);
        assert(!(t1 != t2));
    }
    {
        typedef std::tuple<int, double> T1;
        typedef std::tuple<double, long> T2;
        const T1 t1(1, 2);
        const T2 t2(1, 3);
        assert(!(t1 == t2));
        assert(t1 != t2);
    }
    {
        typedef std::tuple<int, double> T1;
        typedef std::tuple<double, long> T2;
        const T1 t1(1, 2);
        const T2 t2(1.1, 2);
        assert(!(t1 == t2));
        assert(t1 != t2);
    }
    {
        typedef std::tuple<int, double> T1;
        typedef std::tuple<double, long> T2;
        const T1 t1(1, 2);
        const T2 t2(1.1, 3);
        assert(!(t1 == t2));
        assert(t1 != t2);
    }
    {
        typedef std::tuple<long, int, double> T1;
        typedef std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(1, 2, 3);
        assert(t1 == t2);
        assert(!(t1 != t2));
    }
    {
        typedef std::tuple<long, int, double> T1;
        typedef std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(1.1, 2, 3);
        assert(!(t1 == t2));
        assert(t1 != t2);
    }
    {
        typedef std::tuple<long, int, double> T1;
        typedef std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(1, 3, 3);
        assert(!(t1 == t2));
        assert(t1 != t2);
    }
    {
        typedef std::tuple<long, int, double> T1;
        typedef std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(1, 2, 4);
        assert(!(t1 == t2));
        assert(t1 != t2);
    }
    {
        typedef std::tuple<long, int, double> T1;
        typedef std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(1, 3, 2);
        assert(!(t1 == t2));
        assert(t1 != t2);
    }
    {
        typedef std::tuple<long, int, double> T1;
        typedef std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(1.1, 2, 2);
        assert(!(t1 == t2));
        assert(t1 != t2);
    }
    {
        typedef std::tuple<long, int, double> T1;
        typedef std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(1.1, 3, 3);
        assert(!(t1 == t2));
        assert(t1 != t2);
    }
    {
        typedef std::tuple<long, int, double> T1;
        typedef std::tuple<double, long, int> T2;
        const T1 t1(1, 2, 3);
        const T2 t2(1.1, 3, 2);
        assert(!(t1 == t2));
        assert(t1 != t2);
    }
#if TEST_STD_VER > 11
    {
        typedef std::tuple<long, int, double> T1;
        typedef std::tuple<double, long, int> T2;
        constexpr T1 t1(1, 2, 3);
        constexpr T2 t2(1.1, 3, 2);
        static_assert(!(t1 == t2), "");
        static_assert(t1 != t2, "");
    }
#endif

  return 0;
}
