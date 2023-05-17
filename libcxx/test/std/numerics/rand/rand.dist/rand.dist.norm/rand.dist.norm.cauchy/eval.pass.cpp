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
// class cauchy_distribution

// template<class _URNG> result_type operator()(_URNG& g);

#include <random>

#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>

#include "test_macros.h"

double
f(double x, double a, double b)
{
    return 1/3.1415926535897932 * std::atan((x - a)/b) + .5;
}

int main(int, char**)
{
    {
        typedef std::cauchy_distribution<> D;
        typedef std::mt19937 G;
        G g;
        const double a = 10;
        const double b = .5;
        D d(a, b);
        const int N = 1000000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g));
        std::sort(u.begin(), u.end());
        for (int i = 0; i < N; ++i)
            assert(std::abs(f(u[i], a, b) - double(i)/N) < .001);
    }
    {
        typedef std::cauchy_distribution<> D;
        typedef std::mt19937 G;
        G g;
        const double a = -1.5;
        const double b = 1;
        D d(a, b);
        const int N = 1000000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g));
        std::sort(u.begin(), u.end());
        for (int i = 0; i < N; ++i)
            assert(std::abs(f(u[i], a, b) - double(i)/N) < .001);
    }
    {
        typedef std::cauchy_distribution<> D;
        typedef std::mt19937 G;
        G g;
        const double a = .5;
        const double b = 2;
        D d(a, b);
        const int N = 1000000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
            u.push_back(d(g));
        std::sort(u.begin(), u.end());
        for (int i = 0; i < N; ++i)
            assert(std::abs(f(u[i], a, b) - double(i)/N) < .001);
    }

  return 0;
}
