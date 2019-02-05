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
// class discrete_distribution

// template<class _URNG> result_type operator()(_URNG& g, const param_type& parm);

#include <random>
#include <vector>
#include <cassert>

int main()
{
    {
        typedef std::discrete_distribution<> D;
        typedef D::param_type P;
        typedef std::minstd_rand G;
        G g;
        D d;
        double p0[] = {.3, .1, .6};
        P p(p0, p0+3);
        const int N = 10000000;
        std::vector<D::result_type> u(3);
        for (int i = 0; i < N; ++i)
        {
            D::result_type v = d(g, p);
            assert(0 <= v && v <= 2);
            u[v]++;
        }
        std::vector<double> prob = p.probabilities();
        for (int i = 0; i <= 2; ++i)
            assert(std::abs((double)u[i]/N - prob[i]) / prob[i] < 0.001);
    }
}
