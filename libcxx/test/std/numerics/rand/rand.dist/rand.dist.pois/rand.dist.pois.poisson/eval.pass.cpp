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
// class poisson_distribution

// template<class _URNG> result_type operator()(_URNG& g);

#include <random>
#include <cassert>
#include <vector>
#include <numeric>

#include "test_macros.h"

template <class T>
T sqr(T x) {
  return x * x;
}

void test_bad_ranges() {
  // Test cases where the mean is around the largest representable integer for
  // `result_type`. These cases don't generate valid poisson distributions, but
  // at least they don't blow up.
  std::mt19937 eng;

  {
    std::poisson_distribution<std::int16_t> distribution(32710.9);
    for (int i=0; i < 1000; ++i) {
      volatile std::int16_t res = distribution(eng);
      ((void)res);
    }
  }
  {
    std::poisson_distribution<std::int16_t> distribution(std::numeric_limits<std::int16_t>::max());
    for (int i=0; i < 1000; ++i) {
      volatile std::int16_t res = distribution(eng);
      ((void)res);
    }
  }
  {
    std::poisson_distribution<std::int16_t> distribution(
    static_cast<double>(std::numeric_limits<std::int16_t>::max()) + 10);
    for (int i=0; i < 1000; ++i) {
      volatile std::int16_t res = distribution(eng);
      ((void)res);
    }
  }
  {
    std::poisson_distribution<std::int16_t> distribution(
      static_cast<double>(std::numeric_limits<std::int16_t>::max()) * 2);
      for (int i=0; i < 1000; ++i) {
        volatile std::int16_t res = distribution(eng);
        ((void)res);
      }
  }
  {
    // We convert `INF` to `DBL_MAX` otherwise the distribution will hang.
    std::poisson_distribution<std::int16_t> distribution(std::numeric_limits<double>::infinity());
    for (int i=0; i < 1000; ++i) {
      volatile std::int16_t res = distribution(eng);
      ((void)res);
    }
  }
  {
    std::poisson_distribution<std::int16_t> distribution(0);
    for (int i=0; i < 1000; ++i) {
      volatile std::int16_t res = distribution(eng);
      ((void)res);
    }
  }
  {
    // We convert `INF` to `DBL_MAX` otherwise the distribution will hang.
    std::poisson_distribution<std::int16_t> distribution(-100);
    for (int i=0; i < 1000; ++i) {
      volatile std::int16_t res = distribution(eng);
      ((void)res);
    }
  }
}

template <class T>
void tests() {
  {
    typedef std::poisson_distribution<T> D;
    typedef std::minstd_rand G;
    G g;
    D d(2);
    const int N = 100000;
    std::vector<double> u;
    for (int i = 0; i < N; ++i)
    {
      typename D::result_type v = d(g);
      assert(d.min() <= v && v <= d.max());
      u.push_back(v);
    }
    double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
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
    double x_mean = d.mean();
    double x_var = d.mean();
    double x_skew = 1 / std::sqrt(x_var);
    double x_kurtosis = 1 / x_var;
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.03);
  }
  {
    typedef std::poisson_distribution<T> D;
    typedef std::minstd_rand G;
    G g;
    D d(0.75);
    const int N = 100000;
    std::vector<double> u;
    for (int i = 0; i < N; ++i)
    {
      typename D::result_type v = d(g);
      assert(d.min() <= v && v <= d.max());
      u.push_back(v);
    }
    double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
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
    double x_mean = d.mean();
    double x_var = d.mean();
    double x_skew = 1 / std::sqrt(x_var);
    double x_kurtosis = 1 / x_var;
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.04);
  }
  {
    typedef std::poisson_distribution<T> D;
    typedef std::mt19937 G;
    G g;
    D d(20);
    const int N = 1000000;
    std::vector<double> u;
    for (int i = 0; i < N; ++i)
    {
      typename D::result_type v = d(g);
      assert(d.min() <= v && v <= d.max());
      u.push_back(v);
    }
    double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
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
    double x_mean = d.mean();
    double x_var = d.mean();
    double x_skew = 1 / std::sqrt(x_var);
    double x_kurtosis = 1 / x_var;
    assert(std::abs((mean - x_mean) / x_mean) < 0.01);
    assert(std::abs((var - x_var) / x_var) < 0.01);
    assert(std::abs((skew - x_skew) / x_skew) < 0.01);
    assert(std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.01);
  }
}

int main(int, char**) {
  test_bad_ranges();

  tests<short>();
  tests<int>();
  tests<long>();
  tests<long long>();

  tests<unsigned short>();
  tests<unsigned int>();
  tests<unsigned long>();
  tests<unsigned long long>();

#if defined(_LIBCPP_VERSION) // extension
  // TODO: std::poisson_distribution currently doesn't work reliably with small types.
  // tests<int8_t>();
  // tests<uint8_t>();
#if !defined(TEST_HAS_NO_INT128)
  tests<__int128_t>();
  tests<__uint128_t>();
#endif
#endif

  return 0;
}
