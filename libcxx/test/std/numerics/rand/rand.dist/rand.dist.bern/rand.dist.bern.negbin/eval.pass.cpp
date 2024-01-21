//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: long_tests

// This test is super slow, in particular with msan or tsan. In order to avoid timeouts and to
// spend less time waiting for this particular test to complete we compile with optimizations.
// ADDITIONAL_COMPILE_FLAGS(msan): -O1
// ADDITIONAL_COMPILE_FLAGS(tsan): -O1

// FIXME: This and other tests fail under GCC with optimizations enabled.
// More investigation is needed, but it appears that  GCC is performing more constant folding.

// <random>

// template<class IntType = int>
// class negative_binomial_distribution

// template<class _URNG> result_type operator()(_URNG& g);

#include <random>
#include <numeric>
#include <vector>
#include <cassert>

#include "test_macros.h"

template <class T>
T sqr(T x) {
    return x * x;
}

template <class T>
void test1() {
    typedef std::negative_binomial_distribution<T> D;
    typedef std::minstd_rand G;
    G g;
    D d(5, .25);
    const int N = 1000000;
    std::vector<typename D::result_type> u;
    for (int i = 0; i < N; ++i)
    {
        typename D::result_type v = d(g);
        assert(d.min() <= v && v <= d.max());
        u.push_back(v);
    }
    double mean = std::accumulate(u.begin(), u.end(),
                                          double(0)) / u.size();
    double var = 0;
    double skew = 0;
    double kurtosis = 0;
    for (unsigned i = 0; i < u.size(); ++i)
    {
        double dbl = (u[i] - mean);
        double d2 = sqr(dbl);
        var += d2;
        skew += dbl * d2;
        kurtosis += d2 * d2;
    }
    var /= u.size();
    double dev = std::sqrt(var);
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;
    kurtosis -= 3;
    double x_mean = d.k() * (1 - d.p()) / d.p();
    double x_var = x_mean / d.p();
    double x_skew = (2 - d.p()) / std::sqrt(d.k() * (1 - d.p()));
    double x_kurtosis = 6. / d.k() + sqr(d.p()) / (d.k() * (1 - d.p()));
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.02);
}

template <class T>
void test2() {
    typedef std::negative_binomial_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(30, .03125);
    const int N = 1000000;
    std::vector<typename D::result_type> u;
    for (int i = 0; i < N; ++i)
    {
        typename D::result_type v = d(g);
        assert(d.min() <= v && v <= d.max());
        u.push_back(v);
    }
    double mean = std::accumulate(u.begin(), u.end(),
                                          double(0)) / u.size();
    double var = 0;
    double skew = 0;
    double kurtosis = 0;
    for (unsigned i = 0; i < u.size(); ++i)
    {
        double dbl = (u[i] - mean);
        double d2 = sqr(dbl);
        var += d2;
        skew += dbl * d2;
        kurtosis += d2 * d2;
    }
    var /= u.size();
    double dev = std::sqrt(var);
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;
    kurtosis -= 3;
    double x_mean = d.k() * (1 - d.p()) / d.p();
    double x_var = x_mean / d.p();
    double x_skew = (2 - d.p()) / std::sqrt(d.k() * (1 - d.p()));
    double x_kurtosis = 6. / d.k() + sqr(d.p()) / (d.k() * (1 - d.p()));
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.01);
}

template <class T>
void test3() {
    typedef std::negative_binomial_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(40, .25);
    const int N = 1000000;
    std::vector<typename D::result_type> u;
    for (int i = 0; i < N; ++i)
    {
        typename D::result_type v = d(g);
        assert(d.min() <= v && v <= d.max());
        u.push_back(v);
    }
    double mean = std::accumulate(u.begin(), u.end(),
                                          double(0)) / u.size();
    double var = 0;
    double skew = 0;
    double kurtosis = 0;
    for (unsigned i = 0; i < u.size(); ++i)
    {
        double dbl = (u[i] - mean);
        double d2 = sqr(dbl);
        var += d2;
        skew += dbl * d2;
        kurtosis += d2 * d2;
    }
    var /= u.size();
    double dev = std::sqrt(var);
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;
    kurtosis -= 3;
    double x_mean = d.k() * (1 - d.p()) / d.p();
    double x_var = x_mean / d.p();
    double x_skew = (2 - d.p()) / std::sqrt(d.k() * (1 - d.p()));
    double x_kurtosis = 6. / d.k() + sqr(d.p()) / (d.k() * (1 - d.p()));
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.03);
}

template <class T>
void test4() {
    typedef std::negative_binomial_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(40, 1);
    const int N = 1000;
    std::vector<typename D::result_type> u;
    for (int i = 0; i < N; ++i)
    {
        typename D::result_type v = d(g);
        assert(d.min() <= v && v <= d.max());
        u.push_back(v);
    }
    double mean = std::accumulate(u.begin(), u.end(),
                                          double(0)) / u.size();
    double var = 0;
    double skew = 0;
    double kurtosis = 0;
    for (unsigned i = 0; i < u.size(); ++i)
    {
        double dbl = (u[i] - mean);
        double d2 = sqr(dbl);
        var += d2;
        skew += dbl * d2;
        kurtosis += d2 * d2;
    }
    var /= u.size();
    double dev = std::sqrt(var);
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;
    kurtosis -= 3;
    double x_mean = d.k() * (1 - d.p()) / d.p();
    double x_var = x_mean / d.p();
    double x_skew = (2 - d.p()) / std::sqrt(d.k() * (1 - d.p()));
    double x_kurtosis = 6. / d.k() + sqr(d.p()) / (d.k() * (1 - d.p()));
    assert(mean == x_mean);
    assert(var == x_var);
    // assert(skew == x_skew);
    (void)skew; (void)x_skew;
    // assert(kurtosis == x_kurtosis);
    (void)kurtosis; (void)x_kurtosis;
}

template <class T>
void test5() {
    typedef std::negative_binomial_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(127, 0.5);
    const int N = 1000000;
    std::vector<typename D::result_type> u;
    for (int i = 0; i < N; ++i)
    {
        typename D::result_type v = d(g);
        assert(d.min() <= v && v <= d.max());
        u.push_back(v);
    }
    double mean = std::accumulate(u.begin(), u.end(),
                                          double(0)) / u.size();
    double var = 0;
    double skew = 0;
    double kurtosis = 0;
    for (unsigned i = 0; i < u.size(); ++i)
    {
        double dbl = (u[i] - mean);
        double d2 = sqr(dbl);
        var += d2;
        skew += dbl * d2;
        kurtosis += d2 * d2;
    }
    var /= u.size();
    double dev = std::sqrt(var);
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;
    kurtosis -= 3;
    double x_mean = d.k() * (1 - d.p()) / d.p();
    double x_var = x_mean / d.p();
    double x_skew = (2 - d.p()) / std::sqrt(d.k() * (1 - d.p()));
    double x_kurtosis = 6. / d.k() + sqr(d.p()) / (d.k() * (1 - d.p()));
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.04);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.05);
}

template <class T>
void test6() {
    typedef std::negative_binomial_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(1, 0.05);
    const int N = 1000000;
    std::vector<typename D::result_type> u;
    for (int i = 0; i < N; ++i)
    {
        typename D::result_type v = d(g);
        assert(d.min() <= v && v <= d.max());
        u.push_back(v);
    }
    double mean = std::accumulate(u.begin(), u.end(),
                                          double(0)) / u.size();
    double var = 0;
    double skew = 0;
    double kurtosis = 0;
    for (unsigned i = 0; i < u.size(); ++i)
    {
        double dbl = (u[i] - mean);
        double d2 = sqr(dbl);
        var += d2;
        skew += dbl * d2;
        kurtosis += d2 * d2;
    }
    var /= u.size();
    double dev = std::sqrt(var);
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;
    kurtosis -= 3;
    double x_mean = d.k() * (1 - d.p()) / d.p();
    double x_var = x_mean / d.p();
    double x_skew = (2 - d.p()) / std::sqrt(d.k() * (1 - d.p()));
    double x_kurtosis = 6. / d.k() + sqr(d.p()) / (d.k() * (1 - d.p()));
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.03);
}

template <class T>
void tests() {
    test1<T>();
    test2<T>();
    test3<T>();
    test4<T>();
    test5<T>();
    test6<T>();
}

int main(int, char**) {
    tests<short>();
    tests<int>();
    tests<long>();
    tests<long long>();

    tests<unsigned short>();
    tests<unsigned int>();
    tests<unsigned long>();
    tests<unsigned long long>();

#if defined(_LIBCPP_VERSION) // extension
    // TODO: std::negative_binomial_distribution currently doesn't work reliably with small types.
    // tests<int8_t>();
    // tests<uint8_t>();
#if !defined(TEST_HAS_NO_INT128)
    tests<__int128_t>();
    tests<__uint128_t>();
#endif
#endif

    return 0;
}
