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

// template<class _URNG> result_type operator()(_URNG& g);

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include "test_macros.h"

template <class T>
void tests() {
    typedef long long Frequency;
    {
        typedef std::discrete_distribution<T> D;
        typedef std::minstd_rand G;
        G g;
        D d;
        const int N = 100;
        std::vector<Frequency> u(static_cast<std::size_t>(d.max()+1));
        assert(u.max_size() > static_cast<unsigned long long>(d.max()));
        for (int i = 0; i < N; ++i)
        {
            typename D::result_type v = d(g);
            assert(d.min() <= v && v <= d.max());
            u[static_cast<std::size_t>(v)]++;
        }
        std::vector<double> prob = d.probabilities();
        for (unsigned i = 0; i < u.size(); ++i)
            assert((double)u[i]/N == prob[i]);
    }
    {
        typedef std::discrete_distribution<T> D;
        typedef std::minstd_rand G;
        G g;
        double p0[] = {.3};
        D d(p0, p0+1);
        const int N = 100;
        std::vector<Frequency> u(static_cast<std::size_t>(d.max()+1));
        assert(u.max_size() > static_cast<unsigned long long>(d.max()));
        for (int i = 0; i < N; ++i)
        {
            typename D::result_type v = d(g);
            assert(d.min() <= v && v <= d.max());
            u[static_cast<std::size_t>(v)]++;
        }
        std::vector<double> prob = d.probabilities();
        for (unsigned i = 0; i < u.size(); ++i)
            assert((double)u[i]/N == prob[i]);
    }
    {
        typedef std::discrete_distribution<T> D;
        typedef std::minstd_rand G;
        G g;
        double p0[] = {.75, .25};
        D d(p0, p0+2);
        const int N = 1000000;
        std::vector<Frequency> u(static_cast<std::size_t>(d.max()+1));
        assert(u.max_size() > static_cast<unsigned long long>(d.max()));
        for (int i = 0; i < N; ++i)
        {
            typename D::result_type v = d(g);
            assert(d.min() <= v && v <= d.max());
            u[static_cast<std::size_t>(v)]++;
        }
        std::vector<double> prob = d.probabilities();
        for (unsigned i = 0; i < u.size(); ++i)
            assert(std::abs((double)u[i]/N - prob[i]) / prob[i] < 0.001);
    }
    {
        typedef std::discrete_distribution<T> D;
        typedef std::minstd_rand G;
        G g;
        double p0[] = {0, 1};
        D d(p0, p0+2);
        const int N = 1000000;
        std::vector<Frequency> u(static_cast<std::size_t>(d.max()+1));
        assert(u.max_size() > static_cast<unsigned long long>(d.max()));
        for (int i = 0; i < N; ++i)
        {
            typename D::result_type v = d(g);
            assert(d.min() <= v && v <= d.max());
            u[static_cast<std::size_t>(v)]++;
        }
        std::vector<double> prob = d.probabilities();
        assert((double)u[0]/N == prob[0]);
        assert((double)u[1]/N == prob[1]);
    }
    {
        typedef std::discrete_distribution<T> D;
        typedef std::minstd_rand G;
        G g;
        double p0[] = {1, 0};
        D d(p0, p0+2);
        const int N = 1000000;
        std::vector<Frequency> u(static_cast<std::size_t>(d.max()+1));
        assert(u.max_size() > static_cast<unsigned long long>(d.max()));
        for (int i = 0; i < N; ++i)
        {
            typename D::result_type v = d(g);
            assert(d.min() <= v && v <= d.max());
            u[static_cast<std::size_t>(v)]++;
        }
        std::vector<double> prob = d.probabilities();
        assert((double)u[0]/N == prob[0]);
        assert((double)u[1]/N == prob[1]);
    }
    {
        typedef std::discrete_distribution<T> D;
        typedef std::minstd_rand G;
        G g;
        double p0[] = {.3, .1, .6};
        D d(p0, p0+3);
        const int N = 10000000;
        std::vector<Frequency> u(static_cast<std::size_t>(d.max()+1));
        assert(u.max_size() > static_cast<unsigned long long>(d.max()));
        for (int i = 0; i < N; ++i)
        {
            typename D::result_type v = d(g);
            assert(d.min() <= v && v <= d.max());
            u[static_cast<std::size_t>(v)]++;
        }
        std::vector<double> prob = d.probabilities();
        for (unsigned i = 0; i < u.size(); ++i)
            assert(std::abs((double)u[i]/N - prob[i]) / prob[i] < 0.001);
    }
    {
        typedef std::discrete_distribution<T> D;
        typedef std::minstd_rand G;
        G g;
        double p0[] = {0, 25, 75};
        D d(p0, p0+3);
        const int N = 1000000;
        std::vector<Frequency> u(static_cast<std::size_t>(d.max()+1));
        assert(u.max_size() > static_cast<unsigned long long>(d.max()));
        for (int i = 0; i < N; ++i)
        {
            typename D::result_type v = d(g);
            assert(d.min() <= v && v <= d.max());
            u[static_cast<std::size_t>(v)]++;
        }
        std::vector<double> prob = d.probabilities();
        for (unsigned i = 0; i < u.size(); ++i)
            if (prob[i] != 0)
                assert(std::abs((double)u[i]/N - prob[i]) / prob[i] < 0.001);
            else
                assert(u[i] == 0);
    }
    {
        typedef std::discrete_distribution<T> D;
        typedef std::minstd_rand G;
        G g;
        double p0[] = {25, 0, 75};
        D d(p0, p0+3);
        const int N = 1000000;
        std::vector<Frequency> u(static_cast<std::size_t>(d.max()+1));
        assert(u.max_size() > static_cast<unsigned long long>(d.max()));
        for (int i = 0; i < N; ++i)
        {
            typename D::result_type v = d(g);
            assert(d.min() <= v && v <= d.max());
            u[static_cast<std::size_t>(v)]++;
        }
        std::vector<double> prob = d.probabilities();
        for (unsigned i = 0; i < u.size(); ++i)
            if (prob[i] != 0)
                assert(std::abs((double)u[i]/N - prob[i]) / prob[i] < 0.001);
            else
                assert(u[i] == 0);
    }
    {
        typedef std::discrete_distribution<T> D;
        typedef std::minstd_rand G;
        G g;
        double p0[] = {25, 75, 0};
        D d(p0, p0+3);
        const int N = 1000000;
        std::vector<Frequency> u(static_cast<std::size_t>(d.max()+1));
        assert(u.max_size() > static_cast<unsigned long long>(d.max()));
        for (int i = 0; i < N; ++i)
        {
            typename D::result_type v = d(g);
            assert(d.min() <= v && v <= d.max());
            u[static_cast<std::size_t>(v)]++;
        }
        std::vector<double> prob = d.probabilities();
        for (unsigned i = 0; i < u.size(); ++i)
            if (prob[i] != 0)
                assert(std::abs((double)u[i]/N - prob[i]) / prob[i] < 0.001);
            else
                assert(u[i] == 0);
    }
    {
        typedef std::discrete_distribution<T> D;
        typedef std::minstd_rand G;
        G g;
        double p0[] = {0, 0, 1};
        D d(p0, p0+3);
        const int N = 100;
        std::vector<Frequency> u(static_cast<std::size_t>(d.max()+1));
        assert(u.max_size() > static_cast<unsigned long long>(d.max()));
        for (int i = 0; i < N; ++i)
        {
            typename D::result_type v = d(g);
            assert(d.min() <= v && v <= d.max());
            u[static_cast<std::size_t>(v)]++;
        }
        std::vector<double> prob = d.probabilities();
        for (unsigned i = 0; i < u.size(); ++i)
            if (prob[i] != 0)
                assert(std::abs((double)u[i]/N - prob[i]) / prob[i] < 0.001);
            else
                assert(u[i] == 0);
    }
    {
        typedef std::discrete_distribution<T> D;
        typedef std::minstd_rand G;
        G g;
        double p0[] = {0, 1, 0};
        D d(p0, p0+3);
        const int N = 100;
        std::vector<Frequency> u(static_cast<std::size_t>(d.max()+1));
        assert(u.max_size() > static_cast<unsigned long long>(d.max()));
        for (int i = 0; i < N; ++i)
        {
            typename D::result_type v = d(g);
            assert(d.min() <= v && v <= d.max());
            u[static_cast<std::size_t>(v)]++;
        }
        std::vector<double> prob = d.probabilities();
        for (unsigned i = 0; i < u.size(); ++i)
            if (prob[i] != 0)
                assert(std::abs((double)u[i]/N - prob[i]) / prob[i] < 0.001);
            else
                assert(u[i] == 0);
    }
    {
        typedef std::discrete_distribution<T> D;
        typedef std::minstd_rand G;
        G g;
        double p0[] = {1, 0, 0};
        D d(p0, p0+3);
        const int N = 100;
        std::vector<Frequency> u(static_cast<std::size_t>(d.max()+1));
        assert(u.max_size() > static_cast<unsigned long long>(d.max()));
        for (int i = 0; i < N; ++i)
        {
            typename D::result_type v = d(g);
            assert(d.min() <= v && v <= d.max());
            u[static_cast<std::size_t>(v)]++;
        }
        std::vector<double> prob = d.probabilities();
        for (unsigned i = 0; i < u.size(); ++i)
            if (prob[i] != 0)
                assert(std::abs((double)u[i]/N - prob[i]) / prob[i] < 0.001);
            else
                assert(u[i] == 0);
    }
    {
        typedef std::discrete_distribution<T> D;
        typedef std::minstd_rand G;
        G g;
        double p0[] = {33, 0, 0, 67};
        D d(p0, p0+3);
        const int N = 1000000;
        std::vector<Frequency> u(static_cast<std::size_t>(d.max()+1));
        assert(u.max_size() > static_cast<unsigned long long>(d.max()));
        for (int i = 0; i < N; ++i)
        {
            typename D::result_type v = d(g);
            assert(d.min() <= v && v <= d.max());
            u[static_cast<std::size_t>(v)]++;
        }
        std::vector<double> prob = d.probabilities();
        for (unsigned i = 0; i < u.size(); ++i)
            if (prob[i] != 0)
                assert(std::abs((double)u[i]/N - prob[i]) / prob[i] < 0.001);
            else
                assert(u[i] == 0);
    }
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
