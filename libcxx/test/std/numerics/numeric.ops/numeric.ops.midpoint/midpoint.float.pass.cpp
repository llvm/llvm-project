//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <numeric>

// template <class _Float>
// _Tp midpoint(_Float __a, _Float __b) noexcept
//

#include <limits>
#include <numeric>
#include <cassert>

#include "test_macros.h"
#include "fp_compare.h"

//  Totally arbitrary picks for precision
template <typename T>
constexpr T fp_error_pct();

template <>
constexpr float fp_error_pct<float>() { return 1.0e-4f; }

template <>
constexpr double fp_error_pct<double>() { return 1.0e-12; }

template <>
constexpr long double fp_error_pct<long double>() { return 1.0e-13l; }


template <typename T>
void fp_test()
{
    ASSERT_SAME_TYPE(T, decltype(std::midpoint(T(), T())));
    ASSERT_NOEXCEPT(             std::midpoint(T(), T()));

    constexpr T maxV = std::numeric_limits<T>::max();
    constexpr T minV = std::numeric_limits<T>::min();

//  Things that can be compared exactly
    static_assert((std::midpoint(T(0), T(0))   == T(0)),   "");
    static_assert((std::midpoint(T(2), T(4))   == T(3)),   "");
    static_assert((std::midpoint(T(4), T(2))   == T(3)),   "");
    static_assert((std::midpoint(T(3), T(4))   == T(3.5)), "");
    static_assert((std::midpoint(T(0), T(0.4)) == T(0.2)), "");

//  Things that can't be compared exactly
    constexpr T pct = fp_error_pct<T>();
    assert((fptest_close_pct(std::midpoint(T( 1.3), T(11.4)), T( 6.35),    pct)));
    assert((fptest_close_pct(std::midpoint(T(11.33), T(31.45)), T(21.39),  pct)));
    assert((fptest_close_pct(std::midpoint(T(-1.3), T(11.4)), T( 5.05),    pct)));
    assert((fptest_close_pct(std::midpoint(T(11.4), T(-1.3)), T( 5.05),    pct)));
    assert((fptest_close_pct(std::midpoint(T(0.1),  T(0.4)),  T(0.25),     pct)));

    assert((fptest_close_pct(std::midpoint(T(11.2345), T(14.5432)), T(12.88885),  pct)));

//  From e to pi
    assert((fptest_close_pct(std::midpoint(T(2.71828182845904523536028747135266249775724709369995),
                                      T(3.14159265358979323846264338327950288419716939937510)),
                                      T(2.92993724102441923691146542731608269097720824653752),  pct)));

    assert((fptest_close_pct(std::midpoint(maxV, T(0)), maxV/2, pct)));
    assert((fptest_close_pct(std::midpoint(T(0), maxV), maxV/2, pct)));
    assert((fptest_close_pct(std::midpoint(minV, T(0)), minV/2, pct)));
    assert((fptest_close_pct(std::midpoint(T(0), minV), minV/2, pct)));
    assert((fptest_close_pct(std::midpoint(maxV, maxV), maxV,   pct)));
    assert((fptest_close_pct(std::midpoint(minV, minV), minV,   pct)));
    assert((fptest_close_pct(std::midpoint(maxV, minV), maxV/2, pct)));
    assert((fptest_close_pct(std::midpoint(minV, maxV), maxV/2, pct)));

//  Near the min and the max
    assert((fptest_close_pct(std::midpoint(maxV*T(0.75), maxV*T(0.50)),  maxV*T(0.625), pct)));
    assert((fptest_close_pct(std::midpoint(maxV*T(0.50), maxV*T(0.75)),  maxV*T(0.625), pct)));
    assert((fptest_close_pct(std::midpoint(minV*T(2),    minV*T(8)),     minV*T(5),     pct)));

//  Big numbers of different signs
    assert((fptest_close_pct(std::midpoint(maxV*T( 0.75),  maxV*T(-0.5)), maxV*T( 0.125), pct)));
    assert((fptest_close_pct(std::midpoint(maxV*T(-0.75),  maxV*T( 0.5)), maxV*T(-0.125), pct)));

//  Denormalized values
//  TODO

//  Check two values "close to each other"
    T d1 = T(3.14);
    T d0 = std::nextafter(d1, T(2));
    T d2 = std::nextafter(d1, T(5));
    assert(d0 < d1);  // sanity checking
    assert(d1 < d2);  // sanity checking

#if defined(__PPC__) && (defined(__LONG_DOUBLE_128__) && __LONG_DOUBLE_128__) &&                                       \
    !(defined(__LONG_DOUBLE_IEEE128__) && __LONG_DOUBLE_IEEE128__)
//  For 128 bit long double implemented as 2 doubles on PowerPC,
//  nextafterl() of libm gives imprecise results which fails the
//  midpoint() tests below. So skip the test for this case.
    if constexpr (sizeof(T) != 16)
#endif
    {
    //  Since there's nothing in between, the midpoint has to be one or the other
        T res;
        res = std::midpoint(d0, d1);
        assert(res == d0 || res == d1);
        assert(d0 <= res);
        assert(res <= d1);
        res = std::midpoint(d1, d0);
        assert(res == d0 || res == d1);
        assert(d0 <= res);
        assert(res <= d1);

        res = std::midpoint(d1, d2);
        assert(res == d1 || res == d2);
        assert(d1 <= res);
        assert(res <= d2);
        res = std::midpoint(d2, d1);
        assert(res == d1 || res == d2);
        assert(d1 <= res);
        assert(res <= d2);
    }
}


int main (int, char**)
{
    fp_test<float>();
    fp_test<double>();
    fp_test<long double>();

    return 0;
}
