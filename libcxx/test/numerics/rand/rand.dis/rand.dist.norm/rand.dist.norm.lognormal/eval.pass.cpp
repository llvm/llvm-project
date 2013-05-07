//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class lognormal_distribution

// template<class _URNG> result_type operator()(_URNG& g);

#include <random>
#include <cassert>
#include <vector>
#include <numeric>

template <class T>
inline
T
sqr(T x)
{
    return x * x;
}

int main()
{
    {
        typedef std::lognormal_distribution<> D;
        typedef D::param_type P;
        typedef std::mt19937 G;
        G g;
        D d(-1./8192, 0.015625);
        const int N = 1000000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
        {
            D::result_type v = d(g);
            assert(v > 0);
            u.push_back(v);
        }
        double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
        double var = 0;
        double skew = 0;
        double kurtosis = 0;
        for (int i = 0; i < u.size(); ++i)
        {
            double d = (u[i] - mean);
            double d2 = sqr(d);
            var += d2;
            skew += d * d2;
            kurtosis += d2 * d2;
        }
        var /= u.size();
        double dev = std::sqrt(var);
        skew /= u.size() * dev * var;
        kurtosis /= u.size() * var * var;
        kurtosis -= 3;
        double x_mean = std::exp(d.m() + sqr(d.s())/2);
        double x_var = (std::exp(sqr(d.s())) - 1) * std::exp(2*d.m() + sqr(d.s()));
        double x_skew = (std::exp(sqr(d.s())) + 2) *
              std::sqrt((std::exp(sqr(d.s())) - 1));
        double x_kurtosis = std::exp(4*sqr(d.s())) + 2*std::exp(3*sqr(d.s())) +
                          3*std::exp(2*sqr(d.s())) - 6;
        assert(std::abs((mean - x_mean) / x_mean) < 0.01);
        assert(std::abs((var - x_var) / x_var) < 0.01);
        assert(std::abs((skew - x_skew) / x_skew) < 0.05);
        assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.25);
    }
    {
        typedef std::lognormal_distribution<> D;
        typedef D::param_type P;
        typedef std::mt19937 G;
        G g;
        D d(-1./32, 0.25);
        const int N = 1000000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
        {
            D::result_type v = d(g);
            assert(v > 0);
            u.push_back(v);
        }
        double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
        double var = 0;
        double skew = 0;
        double kurtosis = 0;
        for (int i = 0; i < u.size(); ++i)
        {
            double d = (u[i] - mean);
            double d2 = sqr(d);
            var += d2;
            skew += d * d2;
            kurtosis += d2 * d2;
        }
        var /= u.size();
        double dev = std::sqrt(var);
        skew /= u.size() * dev * var;
        kurtosis /= u.size() * var * var;
        kurtosis -= 3;
        double x_mean = std::exp(d.m() + sqr(d.s())/2);
        double x_var = (std::exp(sqr(d.s())) - 1) * std::exp(2*d.m() + sqr(d.s()));
        double x_skew = (std::exp(sqr(d.s())) + 2) *
              std::sqrt((std::exp(sqr(d.s())) - 1));
        double x_kurtosis = std::exp(4*sqr(d.s())) + 2*std::exp(3*sqr(d.s())) +
                          3*std::exp(2*sqr(d.s())) - 6;
        assert(std::abs((mean - x_mean) / x_mean) < 0.01);
        assert(std::abs((var - x_var) / x_var) < 0.01);
        assert(std::abs((skew - x_skew) / x_skew) < 0.01);
        assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.03);
    }
    {
        typedef std::lognormal_distribution<> D;
        typedef D::param_type P;
        typedef std::mt19937 G;
        G g;
        D d(-1./8, 0.5);
        const int N = 1000000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
        {
            D::result_type v = d(g);
            assert(v > 0);
            u.push_back(v);
        }
        double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
        double var = 0;
        double skew = 0;
        double kurtosis = 0;
        for (int i = 0; i < u.size(); ++i)
        {
            double d = (u[i] - mean);
            double d2 = sqr(d);
            var += d2;
            skew += d * d2;
            kurtosis += d2 * d2;
        }
        var /= u.size();
        double dev = std::sqrt(var);
        skew /= u.size() * dev * var;
        kurtosis /= u.size() * var * var;
        kurtosis -= 3;
        double x_mean = std::exp(d.m() + sqr(d.s())/2);
        double x_var = (std::exp(sqr(d.s())) - 1) * std::exp(2*d.m() + sqr(d.s()));
        double x_skew = (std::exp(sqr(d.s())) + 2) *
              std::sqrt((std::exp(sqr(d.s())) - 1));
        double x_kurtosis = std::exp(4*sqr(d.s())) + 2*std::exp(3*sqr(d.s())) +
                          3*std::exp(2*sqr(d.s())) - 6;
        assert(std::abs((mean - x_mean) / x_mean) < 0.01);
        assert(std::abs((var - x_var) / x_var) < 0.01);
        assert(std::abs((skew - x_skew) / x_skew) < 0.02);
        assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.05);
    }
    {
        typedef std::lognormal_distribution<> D;
        typedef D::param_type P;
        typedef std::mt19937 G;
        G g;
        D d;
        const int N = 1000000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
        {
            D::result_type v = d(g);
            assert(v > 0);
            u.push_back(v);
        }
        double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
        double var = 0;
        double skew = 0;
        double kurtosis = 0;
        for (int i = 0; i < u.size(); ++i)
        {
            double d = (u[i] - mean);
            double d2 = sqr(d);
            var += d2;
            skew += d * d2;
            kurtosis += d2 * d2;
        }
        var /= u.size();
        double dev = std::sqrt(var);
        skew /= u.size() * dev * var;
        kurtosis /= u.size() * var * var;
        kurtosis -= 3;
        double x_mean = std::exp(d.m() + sqr(d.s())/2);
        double x_var = (std::exp(sqr(d.s())) - 1) * std::exp(2*d.m() + sqr(d.s()));
        double x_skew = (std::exp(sqr(d.s())) + 2) *
              std::sqrt((std::exp(sqr(d.s())) - 1));
        double x_kurtosis = std::exp(4*sqr(d.s())) + 2*std::exp(3*sqr(d.s())) +
                          3*std::exp(2*sqr(d.s())) - 6;
        assert(std::abs((mean - x_mean) / x_mean) < 0.01);
        assert(std::abs((var - x_var) / x_var) < 0.02);
        assert(std::abs((skew - x_skew) / x_skew) < 0.08);
        assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.4);
    }
    {
        typedef std::lognormal_distribution<> D;
        typedef D::param_type P;
        typedef std::mt19937 G;
        G g;
        D d(-0.78125, 1.25);
        const int N = 1000000;
        std::vector<D::result_type> u;
        for (int i = 0; i < N; ++i)
        {
            D::result_type v = d(g);
            assert(v > 0);
            u.push_back(v);
        }
        double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
        double var = 0;
        double skew = 0;
        double kurtosis = 0;
        for (int i = 0; i < u.size(); ++i)
        {
            double d = (u[i] - mean);
            double d2 = sqr(d);
            var += d2;
            skew += d * d2;
            kurtosis += d2 * d2;
        }
        var /= u.size();
        double dev = std::sqrt(var);
        skew /= u.size() * dev * var;
        kurtosis /= u.size() * var * var;
        kurtosis -= 3;
        double x_mean = std::exp(d.m() + sqr(d.s())/2);
        double x_var = (std::exp(sqr(d.s())) - 1) * std::exp(2*d.m() + sqr(d.s()));
        double x_skew = (std::exp(sqr(d.s())) + 2) *
              std::sqrt((std::exp(sqr(d.s())) - 1));
        double x_kurtosis = std::exp(4*sqr(d.s())) + 2*std::exp(3*sqr(d.s())) +
                          3*std::exp(2*sqr(d.s())) - 6;
        assert(std::abs((mean - x_mean) / x_mean) < 0.01);
        assert(std::abs((var - x_var) / x_var) < 0.04);
        assert(std::abs((skew - x_skew) / x_skew) < 0.2);
        assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.7);
    }
}
