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

// template<class IntType = int>
// class geometric_distribution

// template<class _URNG> result_type operator()(_URNG& g);

#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "test_macros.h"

template <class T>
T sqr(T x) {
    return x * x;
}

void test_small_inputs() {
  std::mt19937 engine;
  std::geometric_distribution<std::int16_t> distribution(5.45361e-311);
  typedef std::geometric_distribution<std::int16_t>::result_type result_type;
  for (int i = 0; i < 1000; ++i) {
    volatile result_type res = distribution(engine);
    ((void)res);
  }
}

template <class T>
void test1() {
    typedef std::geometric_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(.03125);
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
    double x_mean = (1 - d.p()) / d.p();
    double x_var = x_mean / d.p();
    double x_skew = (2 - d.p()) / std::sqrt((1 - d.p()));
    double x_kurtosis = 6 + sqr(d.p()) / (1 - d.p());
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.01);
}

template <class T>
void test2() {
    typedef std::geometric_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(0.05);
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
    double x_mean = (1 - d.p()) / d.p();
    double x_var = x_mean / d.p();
    double x_skew = (2 - d.p()) / std::sqrt((1 - d.p()));
    double x_kurtosis = 6 + sqr(d.p()) / (1 - d.p());
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.03);
}

template <class T>
void test3() {
    typedef std::geometric_distribution<T> D;
    typedef std::minstd_rand G;
    G g;
    D d(.25);
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
    double x_mean = (1 - d.p()) / d.p();
    double x_var = x_mean / d.p();
    double x_skew = (2 - d.p()) / std::sqrt((1 - d.p()));
    double x_kurtosis = 6 + sqr(d.p()) / (1 - d.p());
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.02);
}

template <class T>
void test4() {
    typedef std::geometric_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(0.5);
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
    double x_mean = (1 - d.p()) / d.p();
    double x_var = x_mean / d.p();
    double x_skew = (2 - d.p()) / std::sqrt((1 - d.p()));
    double x_kurtosis = 6 + sqr(d.p()) / (1 - d.p());
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.02);
}

template <class T>
void test5() {
    typedef std::geometric_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(0.75);
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
    double x_mean = (1 - d.p()) / d.p();
    double x_var = x_mean / d.p();
    double x_skew = (2 - d.p()) / std::sqrt((1 - d.p()));
    double x_kurtosis = 6 + sqr(d.p()) / (1 - d.p());
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.02);
}

template <class T>
void test6() {
    typedef std::geometric_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(0.96875);
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
    double x_mean = (1 - d.p()) / d.p();
    double x_var = x_mean / d.p();
    double x_skew = (2 - d.p()) / std::sqrt((1 - d.p()));
    double x_kurtosis = 6 + sqr(d.p()) / (1 - d.p());
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.02);
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
    test_small_inputs();

    tests<short>();
    tests<int>();
    tests<long>();
    tests<long long>();

    tests<unsigned short>();
    tests<unsigned int>();
    tests<unsigned long>();
    tests<unsigned long long>();

#if defined(_LIBCPP_VERSION) // extension
    // TODO: std::geometric_distribution currently doesn't work reliably with small types.
    // tests<int8_t>();
    // tests<uint8_t>();
#if !defined(TEST_HAS_NO_INT128)
    tests<__int128_t>();
    tests<__uint128_t>();
#endif
#endif

    return 0;
}
