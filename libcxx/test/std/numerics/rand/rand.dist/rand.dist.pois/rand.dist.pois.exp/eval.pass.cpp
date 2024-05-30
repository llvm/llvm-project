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

// template<class RealType = double>
// class exponential_distribution

// template<class _URNG> result_type operator()(_URNG& g);

#include <random>
#include <cassert>
#include <vector>
#include <numeric>
#include <cstddef>

#include "test_macros.h"

template <class T>
inline
T
sqr(T x)
{
    return x * x;
}

int main(int, char**)
{
    {
        typedef std::exponential_distribution<> D;
        typedef std::mt19937 G;
        G g;
        D d(.75);
        const int N = 1000000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
        {
            D::result_type v = d(g);
            assert(d.min() < v);
            u.push_back(v);
        }
        double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
        double var = 0;
        double skew = 0;
        double kurtosis = 0;
        for (std::size_t i = 0; i < u.size(); ++i)
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
        double x_mean = 1/d.lambda();
        double x_var = 1/sqr(d.lambda());
        double x_skew = 2;
        double x_kurtosis = 6;
        assert(std::abs((mean - x_mean) / x_mean) < 0.01);
        assert(std::abs((var - x_var) / x_var) < 0.01);
        assert(std::abs((skew - x_skew) / x_skew) < 0.01);
        assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.03);
    }
    {
        typedef std::exponential_distribution<> D;
        typedef std::mt19937 G;
        G g;
        D d(1);
        const int N = 1000000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
        {
            D::result_type v = d(g);
            assert(d.min() < v);
            u.push_back(v);
        }
        double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
        double var = 0;
        double skew = 0;
        double kurtosis = 0;
        for (std::size_t i = 0; i < u.size(); ++i)
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
        double x_mean = 1/d.lambda();
        double x_var = 1/sqr(d.lambda());
        double x_skew = 2;
        double x_kurtosis = 6;
        assert(std::abs((mean - x_mean) / x_mean) < 0.01);
        assert(std::abs((var - x_var) / x_var) < 0.01);
        assert(std::abs((skew - x_skew) / x_skew) < 0.01);
        assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.03);
    }
    {
        typedef std::exponential_distribution<> D;
        typedef std::mt19937 G;
        G g;
        D d(10);
        const int N = 1000000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
        {
            D::result_type v = d(g);
            assert(d.min() < v);
            u.push_back(v);
        }
        double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
        double var = 0;
        double skew = 0;
        double kurtosis = 0;
        for (std::size_t i = 0; i < u.size(); ++i)
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
        double x_mean = 1/d.lambda();
        double x_var = 1/sqr(d.lambda());
        double x_skew = 2;
        double x_kurtosis = 6;
        assert(std::abs((mean - x_mean) / x_mean) < 0.01);
        assert(std::abs((var - x_var) / x_var) < 0.01);
        assert(std::abs((skew - x_skew) / x_skew) < 0.01);
        assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.03);
    }

  return 0;
}
