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
// class piecewise_linear_distribution

// template<class _URNG> result_type operator()(_URNG& g);

#include <random>
#include <algorithm>
#include <vector>
#include <iterator>
#include <numeric>
#include <cassert>
#include <limits>

#include "test_macros.h"

template <class T>
inline
T
sqr(T x)
{
    return x*x;
}

double
f(double x, double a, double m, double b, double c)
{
    return a + m*(sqr(x) - sqr(b))/2 + c*(x-b);
}

void
test1()
{
    typedef std::piecewise_linear_distribution<> D;
    typedef std::mt19937_64 G;
    G g;
    double b[] = {10, 14, 16, 17};
    double p[] = {0, 1, 1, 0};
    const std::size_t Np = sizeof(p) / sizeof(p[0]) - 1;
    D d(b, b+Np+1, p);
    const int N = 1000000;
    std::vector<D::result_type> u;
    for (std::size_t i = 0; i < N; ++i)
    {
        D::result_type v = d(g);
        assert(d.min() <= v && v < d.max());
        u.push_back(v);
    }
    std::sort(u.begin(), u.end());
    std::ptrdiff_t kp = -1;
    double a = std::numeric_limits<double>::quiet_NaN();
    double m = std::numeric_limits<double>::quiet_NaN();
    double bk = std::numeric_limits<double>::quiet_NaN();
    double c = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> areas(Np);
    double S = 0;
    for (std::size_t i = 0; i < areas.size(); ++i)
    {
        areas[i] = (p[i]+p[i+1])*(b[i+1]-b[i])/2;
        S += areas[i];
    }
    for (std::size_t i = 0; i < areas.size(); ++i)
        areas[i] /= S;
    for (std::size_t i = 0; i < Np+1; ++i)
        p[i] /= S;
    for (std::size_t i = 0; i < N; ++i)
    {
      std::ptrdiff_t k = std::lower_bound(b, b + Np + 1, u[i]) - b - 1;
      if (k != kp) {
        a = 0;
        for (int j = 0; j < k; ++j)
          a += areas[j];
        m  = (p[k + 1] - p[k]) / (b[k + 1] - b[k]);
        bk = b[k];
        c  = (b[k + 1] * p[k] - b[k] * p[k + 1]) / (b[k + 1] - b[k]);
        kp = k;
        }
      assert(std::abs(f(u[i], a, m, bk, c) - double(i) / N) < .0013);
    }
}

void
test2()
{
    typedef std::piecewise_linear_distribution<> D;
    typedef std::mt19937_64 G;
    G g;
    double b[] = {10, 14, 16, 17};
    double p[] = {0, 0, 1, 0};
    const std::size_t Np = sizeof(p) / sizeof(p[0]) - 1;
    D d(b, b+Np+1, p);
    const int N = 1000000;
    std::vector<D::result_type> u;
    for (std::size_t i = 0; i < N; ++i)
    {
        D::result_type v = d(g);
        assert(d.min() <= v && v < d.max());
        u.push_back(v);
    }
    std::sort(u.begin(), u.end());
    std::ptrdiff_t kp = -1;
    double a = std::numeric_limits<double>::quiet_NaN();
    double m = std::numeric_limits<double>::quiet_NaN();
    double bk = std::numeric_limits<double>::quiet_NaN();
    double c = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> areas(Np);
    double S = 0;
    for (std::size_t i = 0; i < areas.size(); ++i)
    {
        areas[i] = (p[i]+p[i+1])*(b[i+1]-b[i])/2;
        S += areas[i];
    }
    for (std::size_t i = 0; i < areas.size(); ++i)
        areas[i] /= S;
    for (std::size_t i = 0; i < Np+1; ++i)
        p[i] /= S;
    for (std::size_t i = 0; i < N; ++i)
    {
      std::ptrdiff_t k = std::lower_bound(b, b + Np + 1, u[i]) - b - 1;
      if (k != kp) {
        a = 0;
        for (int j = 0; j < k; ++j)
          a += areas[j];
        m  = (p[k + 1] - p[k]) / (b[k + 1] - b[k]);
        bk = b[k];
        c  = (b[k + 1] * p[k] - b[k] * p[k + 1]) / (b[k + 1] - b[k]);
        kp = k;
        }
      assert(std::abs(f(u[i], a, m, bk, c) - double(i) / N) < .0013);
    }
}

void
test3()
{
    typedef std::piecewise_linear_distribution<> D;
    typedef std::mt19937_64 G;
    G g;
    double b[] = {10, 14, 16, 17};
    double p[] = {1, 0, 0, 0};
    const std::size_t Np = sizeof(p) / sizeof(p[0]) - 1;
    D d(b, b+Np+1, p);
    const std::size_t N = 1000000;
    std::vector<D::result_type> u;
    for (std::size_t i = 0; i < N; ++i)
    {
        D::result_type v = d(g);
        assert(d.min() <= v && v < d.max());
        u.push_back(v);
    }
    std::sort(u.begin(), u.end());
    std::ptrdiff_t kp = -1;
    double a = std::numeric_limits<double>::quiet_NaN();
    double m = std::numeric_limits<double>::quiet_NaN();
    double bk = std::numeric_limits<double>::quiet_NaN();
    double c = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> areas(Np);
    double S = 0;
    for (std::size_t i = 0; i < areas.size(); ++i)
    {
        areas[i] = (p[i]+p[i+1])*(b[i+1]-b[i])/2;
        S += areas[i];
    }
    for (std::size_t i = 0; i < areas.size(); ++i)
        areas[i] /= S;
    for (std::size_t i = 0; i < Np+1; ++i)
        p[i] /= S;
    for (std::size_t i = 0; i < N; ++i)
    {
      std::ptrdiff_t k = std::lower_bound(b, b + Np + 1, u[i]) - b - 1;
      if (k != kp) {
        a = 0;
        for (int j = 0; j < k; ++j)
          a += areas[j];
        m  = (p[k + 1] - p[k]) / (b[k + 1] - b[k]);
        bk = b[k];
        c  = (b[k + 1] * p[k] - b[k] * p[k + 1]) / (b[k + 1] - b[k]);
        kp = k;
        }
      assert(std::abs(f(u[i], a, m, bk, c) - double(i) / N) < .0013);
    }
}

void
test4()
{
    typedef std::piecewise_linear_distribution<> D;
    typedef std::mt19937_64 G;
    G g;
    double b[] = {10, 14, 16};
    double p[] = {0, 1, 0};
    const std::size_t Np = sizeof(p) / sizeof(p[0]) - 1;
    D d(b, b+Np+1, p);
    const int N = 1000000;
    std::vector<D::result_type> u;
    for (std::size_t i = 0; i < N; ++i)
    {
        D::result_type v = d(g);
        assert(d.min() <= v && v < d.max());
        u.push_back(v);
    }
    std::sort(u.begin(), u.end());
    std::ptrdiff_t kp = -1;
    double a = std::numeric_limits<double>::quiet_NaN();
    double m = std::numeric_limits<double>::quiet_NaN();
    double bk = std::numeric_limits<double>::quiet_NaN();
    double c = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> areas(Np);
    double S = 0;
    for (std::size_t i = 0; i < areas.size(); ++i)
    {
        areas[i] = (p[i]+p[i+1])*(b[i+1]-b[i])/2;
        S += areas[i];
    }
    for (std::size_t i = 0; i < areas.size(); ++i)
        areas[i] /= S;
    for (std::size_t i = 0; i < Np+1; ++i)
        p[i] /= S;
    for (std::size_t i = 0; i < N; ++i)
    {
      std::ptrdiff_t k = std::lower_bound(b, b + Np + 1, u[i]) - b - 1;
      if (k != kp) {
        a = 0;
        for (int j = 0; j < k; ++j)
          a += areas[j];
        assert(k < static_cast<int>(Np));
        m  = (p[k + 1] - p[k]) / (b[k + 1] - b[k]);
        bk = b[k];
        c  = (b[k + 1] * p[k] - b[k] * p[k + 1]) / (b[k + 1] - b[k]);
        kp = k;
        }
      assert(std::abs(f(u[i], a, m, bk, c) - double(i) / N) < .0013);
    }
}

void
test5()
{
    typedef std::piecewise_linear_distribution<> D;
    typedef std::mt19937_64 G;
    G g;
    double b[] = {10, 14};
    double p[] = {1, 1};
    const std::size_t Np = sizeof(p) / sizeof(p[0]) - 1;
    D d(b, b+Np+1, p);
    const int N = 1000000;
    std::vector<D::result_type> u;
    for (std::size_t i = 0; i < N; ++i)
    {
        D::result_type v = d(g);
        assert(d.min() <= v && v < d.max());
        u.push_back(v);
    }
    std::sort(u.begin(), u.end());
    std::ptrdiff_t kp = -1;
    double a = std::numeric_limits<double>::quiet_NaN();
    double m = std::numeric_limits<double>::quiet_NaN();
    double bk = std::numeric_limits<double>::quiet_NaN();
    double c = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> areas(Np);
    double S = 0;
    for (std::size_t i = 0; i < areas.size(); ++i)
    {
        assert(i < Np);
        areas[i] = (p[i]+p[i+1])*(b[i+1]-b[i])/2;
        S += areas[i];
    }
    for (std::size_t i = 0; i < areas.size(); ++i)
        areas[i] /= S;
    for (std::size_t i = 0; i < Np+1; ++i)
        p[i] /= S;
    for (std::size_t i = 0; i < N; ++i)
    {
      std::ptrdiff_t k = std::lower_bound(b, b + Np + 1, u[i]) - b - 1;
      if (k != kp) {
        a = 0;
        for (int j = 0; j < k; ++j)
          a += areas[j];
        assert(k < static_cast<int>(Np));
        m  = (p[k + 1] - p[k]) / (b[k + 1] - b[k]);
        bk = b[k];
        c  = (b[k + 1] * p[k] - b[k] * p[k + 1]) / (b[k + 1] - b[k]);
        kp = k;
        }
      assert(std::abs(f(u[i], a, m, bk, c) - double(i) / N) < .0013);
    }
}

void
test6()
{
    typedef std::piecewise_linear_distribution<> D;
    typedef std::mt19937_64 G;
    G g;
    double b[] = {10, 14, 16, 17};
    double p[] = {25, 62.5, 12.5, 0};
    const std::size_t Np = sizeof(p) / sizeof(p[0]) - 1;
    D d(b, b+Np+1, p);
    const int N = 1000000;
    std::vector<D::result_type> u;
    for (std::size_t i = 0; i < N; ++i)
    {
        D::result_type v = d(g);
        assert(d.min() <= v && v < d.max());
        u.push_back(v);
    }
    std::sort(u.begin(), u.end());
    std::ptrdiff_t kp = -1;
    double a = std::numeric_limits<double>::quiet_NaN();
    double m = std::numeric_limits<double>::quiet_NaN();
    double bk = std::numeric_limits<double>::quiet_NaN();
    double c = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> areas(Np);
    double S = 0;
    for (std::size_t i = 0; i < areas.size(); ++i)
    {
        areas[i] = (p[i]+p[i+1])*(b[i+1]-b[i])/2;
        S += areas[i];
    }
    for (std::size_t i = 0; i < areas.size(); ++i)
        areas[i] /= S;
    for (std::size_t i = 0; i < Np+1; ++i)
        p[i] /= S;
    for (std::size_t i = 0; i < N; ++i)
    {
      std::ptrdiff_t k = std::lower_bound(b, b + Np + 1, u[i]) - b - 1;
      if (k != kp) {
        a = 0;
        for (int j = 0; j < k; ++j)
          a += areas[j];
        m  = (p[k + 1] - p[k]) / (b[k + 1] - b[k]);
        bk = b[k];
        c  = (b[k + 1] * p[k] - b[k] * p[k + 1]) / (b[k + 1] - b[k]);
        kp = k;
        }
      assert(std::abs(f(u[i], a, m, bk, c) - double(i) / N) < .0013);
    }
}

int main(int, char**)
{
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();

  return 0;
}
