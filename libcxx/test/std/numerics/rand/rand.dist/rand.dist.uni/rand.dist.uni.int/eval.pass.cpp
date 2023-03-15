//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: long_tests

// <random>

// template<class _IntType = int>
// class uniform_int_distribution

// template<class _URNG> result_type operator()(_URNG& g);

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "test_macros.h"

// The __int128 conversions to/from floating point crash on MinGW on x86_64.
// This is fixed in Clang 14 by https://reviews.llvm.org/D110413.
#if defined(__x86_64__) && defined(__MINGW32__) && defined(__clang_major__) && __clang_major__ < 14
 #define TEST_BUGGY_I128_FP
#endif

template <class T>
T sqr(T x) {
    return x * x;
}

template <class ResultType, class EngineType>
void test_statistics(ResultType a, ResultType b) {
    ASSERT_SAME_TYPE(typename std::uniform_int_distribution<ResultType>::result_type, ResultType);

    EngineType g;
    std::uniform_int_distribution<ResultType> dist(a, b);
    assert(dist.a() == a);
    assert(dist.b() == b);
    std::vector<ResultType> u;
    for (int i = 0; i < 10000; ++i) {
        ResultType v = dist(g);
        assert(a <= v && v <= b);
        u.push_back(v);
    }

    // Quick check: The chance of getting *no* hits in any given tenth of the range
    // is (0.9)^10000, or "ultra-astronomically low."
    bool bottom_tenth = false;
    bool top_tenth = false;
    for (std::size_t i = 0; i < u.size(); ++i) {
        bottom_tenth = bottom_tenth || (u[i] <= (a + (b / 10) - (a / 10)));
        top_tenth = top_tenth || (u[i] >= (b - (b / 10) + (a / 10)));
    }
    assert(bottom_tenth);  // ...is populated
    assert(top_tenth);  // ...is populated

    // Now do some more involved statistical math.
    double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
    double var = 0;
    double skew = 0;
    double kurtosis = 0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        double dbl = (u[i] - mean);
        double d2 = dbl * dbl;
        var += d2;
        skew += dbl * d2;
        kurtosis += d2 * d2;
    }
    var /= u.size();
    double dev = std::sqrt(var);
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;

    double expected_mean = double(a) + double(b)/2 - double(a)/2;
    double expected_var = (sqr(double(b) - double(a) + 1) - 1) / 12;

    double range = double(b) - double(a) + 1.0;
    assert(range > range / 10);  // i.e., it's not infinity

    assert(std::abs(mean - expected_mean) < range / 100);
    assert(std::abs(var - expected_var) < expected_var / 50);
    assert(-0.1 < skew && skew < 0.1);
    assert(1.6 < kurtosis && kurtosis < 2.0);
}

template <class ResultType, class EngineType>
void test_statistics() {
    test_statistics<ResultType, EngineType>(0, std::numeric_limits<ResultType>::max());
}

int main(int, char**)
{
    test_statistics<int, std::minstd_rand0>();
    test_statistics<int, std::minstd_rand>();
    test_statistics<int, std::mt19937>();
    test_statistics<int, std::mt19937_64>();
    test_statistics<int, std::ranlux24_base>();
    test_statistics<int, std::ranlux48_base>();
    test_statistics<int, std::ranlux24>();
    test_statistics<int, std::ranlux48>();
    test_statistics<int, std::knuth_b>();
    test_statistics<int, std::minstd_rand0>(-6, 106);
    test_statistics<int, std::minstd_rand>(5, 100);

    test_statistics<short, std::minstd_rand0>();
    test_statistics<int, std::minstd_rand0>();
    test_statistics<long, std::minstd_rand0>();
    test_statistics<long long, std::minstd_rand0>();

    test_statistics<unsigned short, std::minstd_rand0>();
    test_statistics<unsigned int, std::minstd_rand0>();
    test_statistics<unsigned long, std::minstd_rand0>();
    test_statistics<unsigned long long, std::minstd_rand0>();

    test_statistics<short, std::minstd_rand0>(SHRT_MIN, SHRT_MAX);

#if defined(_LIBCPP_VERSION) // extension
    test_statistics<std::int8_t, std::minstd_rand0>();
    test_statistics<std::uint8_t, std::minstd_rand0>();

#if !defined(TEST_HAS_NO_INT128) && !defined(TEST_BUGGY_I128_FP)
    test_statistics<__int128_t, std::minstd_rand0>();
    test_statistics<__uint128_t, std::minstd_rand0>();

    test_statistics<__int128_t, std::minstd_rand0>(-100, 900);
    test_statistics<__int128_t, std::minstd_rand0>(0, UINT64_MAX);
    test_statistics<__int128_t, std::minstd_rand0>(std::numeric_limits<__int128_t>::min(), std::numeric_limits<__int128_t>::max());
    test_statistics<__uint128_t, std::minstd_rand0>(0, UINT64_MAX);
#endif
#endif

    return 0;
}
