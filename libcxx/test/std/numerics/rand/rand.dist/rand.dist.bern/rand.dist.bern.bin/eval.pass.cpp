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
// class binomial_distribution

// template<class _URNG> result_type operator()(_URNG& g);

#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "test_macros.h"

template <class T>
T sqr(T x) {
    return x * x;
}

template <class T>
void test1() {
    typedef std::binomial_distribution<T> D;
    typedef std::mt19937_64 G;
    G g;
    D d(5, .75);
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
    double x_mean = d.t() * d.p();
    double x_var = x_mean*(1-d.p());
    double x_skew = (1-2*d.p()) / std::sqrt(x_var);
    double x_kurtosis = (1-6*d.p()*(1-d.p())) / x_var;
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.08);
}

template <class T>
void test2() {
    typedef std::binomial_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(30, .03125);
    const int N = 100000;
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
    double x_mean = d.t() * d.p();
    double x_var = x_mean*(1-d.p());
    double x_skew = (1-2*d.p()) / std::sqrt(x_var);
    double x_kurtosis = (1-6*d.p()*(1-d.p())) / x_var;
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.02);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.08);
}

template <class T>
void test3() {
    typedef std::binomial_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(40, .25);
    const int N = 100000;
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
    double x_mean = d.t() * d.p();
    double x_var = x_mean*(1-d.p());
    double x_skew = (1-2*d.p()) / std::sqrt(x_var);
    double x_kurtosis = (1-6*d.p()*(1-d.p())) / x_var;
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.07);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 2.0);
}

template <class T>
void test4() {
    typedef std::binomial_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(40, 0);
    const int N = 100000;
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
    // In this case:
    //   skew     computes to 0./0. == nan
    //   kurtosis computes to 0./0. == nan
    //   x_skew     == inf
    //   x_kurtosis == inf
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;
    kurtosis -= 3;
    double x_mean = d.t() * d.p();
    double x_var = x_mean*(1-d.p());
    double x_skew = (1-2*d.p()) / std::sqrt(x_var);
    double x_kurtosis = (1-6*d.p()*(1-d.p())) / x_var;
    assert(mean == x_mean);
    assert(var == x_var);
    // assert(skew == x_skew);
    (void)skew; (void)x_skew;
    // assert(kurtosis == x_kurtosis);
    (void)kurtosis; (void)x_kurtosis;
}

template <class T>
void test5() {
    typedef std::binomial_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(40, 1);
    const int N = 100000;
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
    // In this case:
    //   skew     computes to 0./0. == nan
    //   kurtosis computes to 0./0. == nan
    //   x_skew     == -inf
    //   x_kurtosis == inf
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;
    kurtosis -= 3;
    double x_mean = d.t() * d.p();
    double x_var = x_mean*(1-d.p());
    double x_skew = (1-2*d.p()) / std::sqrt(x_var);
    double x_kurtosis = (1-6*d.p()*(1-d.p())) / x_var;
    assert(mean == x_mean);
    assert(var == x_var);
    // assert(skew == x_skew);
    (void)skew; (void)x_skew;
    // assert(kurtosis == x_kurtosis);
    (void)kurtosis; (void)x_kurtosis;
}

template <class T>
void test6() {
    typedef std::binomial_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(127, 0.5);
    const int N = 100000;
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
    double x_mean = d.t() * d.p();
    double x_var = x_mean*(1-d.p());
    double x_skew = (1-2*d.p()) / std::sqrt(x_var);
    double x_kurtosis = (1-6*d.p()*(1-d.p())) / x_var;
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs(skew - x_skew) < 0.02);
    assert(std::abs(kurtosis - x_kurtosis) < 0.03);
}

template <class T>
void test7() {
    typedef std::binomial_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(1, 0.5);
    const int N = 100000;
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
    double x_mean = d.t() * d.p();
    double x_var = x_mean*(1-d.p());
    double x_skew = (1-2*d.p()) / std::sqrt(x_var);
    double x_kurtosis = (1-6*d.p()*(1-d.p())) / x_var;
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs(skew - x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.01);
}

template <class T>
void test8() {
    const int N = 100000;
    std::mt19937 gen1;
    std::mt19937 gen2;

    using UnsignedT = typename std::make_unsigned<T>::type;
    std::binomial_distribution<T>         dist1(5, 0.1);
    std::binomial_distribution<UnsignedT> dist2(5, 0.1);

    for (int i = 0; i < N; ++i) {
        T r1 = dist1(gen1);
        UnsignedT r2 = dist2(gen2);
        assert(r1 >= 0);
        assert(static_cast<UnsignedT>(r1) == r2);
    }
}

template <class T>
void test9() {
    typedef std::binomial_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(0, 0.005);
    const int N = 100000;
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
    // In this case:
    //   skew     computes to 0./0. == nan
    //   kurtosis computes to 0./0. == nan
    //   x_skew     == inf
    //   x_kurtosis == inf
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;
    kurtosis -= 3;
    double x_mean = d.t() * d.p();
    double x_var = x_mean*(1-d.p());
    double x_skew = (1-2*d.p()) / std::sqrt(x_var);
    double x_kurtosis = (1-6*d.p()*(1-d.p())) / x_var;
    assert(mean == x_mean);
    assert(var == x_var);
    // assert(skew == x_skew);
    (void)skew; (void)x_skew;
    // assert(kurtosis == x_kurtosis);
    (void)kurtosis; (void)x_kurtosis;
}

template <class T>
void test10() {
    typedef std::binomial_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(0, 0);
    const int N = 100000;
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
    // In this case:
    //   skew     computes to 0./0. == nan
    //   kurtosis computes to 0./0. == nan
    //   x_skew     == inf
    //   x_kurtosis == inf
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;
    kurtosis -= 3;
    double x_mean = d.t() * d.p();
    double x_var = x_mean*(1-d.p());
    double x_skew = (1-2*d.p()) / std::sqrt(x_var);
    double x_kurtosis = (1-6*d.p()*(1-d.p())) / x_var;
    assert(mean == x_mean);
    assert(var == x_var);
    // assert(skew == x_skew);
    (void)skew; (void)x_skew;
    // assert(kurtosis == x_kurtosis);
    (void)kurtosis; (void)x_kurtosis;
}

template <class T>
void test11() {
    typedef std::binomial_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(0, 1);
    const int N = 100000;
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
    // In this case:
    //   skew     computes to 0./0. == nan
    //   kurtosis computes to 0./0. == nan
    //   x_skew     == -inf
    //   x_kurtosis == inf
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;
    kurtosis -= 3;
    double x_mean = d.t() * d.p();
    double x_var = x_mean*(1-d.p());
    double x_skew = (1-2*d.p()) / std::sqrt(x_var);
    double x_kurtosis = (1-6*d.p()*(1-d.p())) / x_var;
    assert(mean == x_mean);
    assert(var == x_var);
    // assert(skew == x_skew);
    (void)skew; (void)x_skew;
    // assert(kurtosis == x_kurtosis);
    (void)kurtosis; (void)x_kurtosis;
}

template <class T>
void tests() {
    test1<T>();
    test2<T>();
    test3<T>();
    test4<T>();
    test5<T>();
    test6<T>();
    test7<T>();
    test8<T>();
    test9<T>();
    test10<T>();
    test11<T>();
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
    tests<std::int8_t>();
    tests<std::uint8_t>();
#if !defined(TEST_HAS_NO_INT128)
    tests<__int128_t>();
    tests<__uint128_t>();
#endif
#endif

    return 0;
}
