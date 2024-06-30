//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <cmath>

// double         assoc_laguerre( unsigned n, unsigned m, double      x);
// float          assoc_laguerre( unsigned n, unsigned m, float       x);
// long double    assoc_laguerre( unsigned n, unsigned m, long double x);
// float          assoc_laguerref(unsigned n, unsigned m, float       x);
// long double    assoc_laguerrel(unsigned n, unsigned m, long double x);
// template <class Integer>
// double         assoc_laguerre( unsigned n, unsigned m, Integer     x);

#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>

#include "type_algorithms.h"

inline constexpr unsigned g_max_n = 128;

template <class Real>
std::array<Real, 8> sample_points() {
  return {0.0, 0.1, 0.5, 1.0, 2.7, 5.6, 10.3, 13.0};
}

template <class Real>
class CompareFloatingValues {
private:
  Real tol;

public:
  CompareFloatingValues() {
    if (std::is_same_v<Real, float>)
      tol = 1e-4f;
    else if (std::is_same_v<Real, double>)
      tol = 1e-9;
    else
      tol = 1e-12l;
  }

  bool operator()(Real result, Real expected) const {
    if (std::isinf(expected) && std::isinf(result))
      return result == expected;

    if (std::isnan(expected) || std::isnan(result))
      return false;

    return std::abs(result - expected) < (tol + std::abs(expected) * tol);
  }
};

template <class Real, std::size_t N>
class Polynomial {
  static_assert(N >= 1);
  std::array<int, N> coeffs_;
  int scaling_;

public:
  // Polynomial P = ( a0 + a1*x + a2*x^2 + ... + a_{N-1}*x^{N-1} ) / C
  //    coeffs  = [a0, a1, .., a_{N-1}]
  //    scaling_ = C
  Polynomial(Real, const int (&coeffs)[N], int scaling) : coeffs_{std::to_array(coeffs)}, scaling_{scaling} {}

  // Evaluation at value x via Horner's method.
  Real operator()(Real x) const {
    return std::accumulate(coeffs_.rbegin() + 1,
                           coeffs_.rend(),
                           static_cast<Real>(coeffs_.back()),
                           [x](Real acc, int coeff) -> Real { return acc * x + coeff; }) /
           scaling_;
  };
};

template <class Real, std::size_t N>
Polynomial(Real, const int (&)[N], int) -> Polynomial<Real, N>;

template <class Real>
void test() {
  { // checks if NaNs are reported correctly (i.e. output == input for input == NaN)
    using nl = std::numeric_limits<Real>;
    for (unsigned n = 0; n < g_max_n; ++n)
      for (unsigned m = 0; m < g_max_n; ++m)
        for (Real NaN : {nl::quiet_NaN(), nl::signaling_NaN()})
          assert(std::isnan(std::assoc_laguerre(n, m, NaN)));
  }

  { // simple sample points for n, m = 0..127 should not produce NaNs.
    for (unsigned n = 0; n < g_max_n; ++n)
      for (unsigned m = 0; m < g_max_n; ++m)
        for (Real x : sample_points<Real>())
          assert(!std::isnan(std::assoc_laguerre(n, m, x)));
  }

  { // For any $x < 0$ a domain_error exception should be thrown.
#ifndef _LIBCPP_NO_EXCEPTIONS
    for (unsigned n = 0; n < g_max_n; ++n)
      for (unsigned m = 0; m < g_max_n; ++m) {
        const auto is_domain_error = [n, m](Real x) -> bool {
          try {
            std::assoc_laguerre(n, m, x);
          } catch (const std::domain_error&) {
            return true;
          }
          return false;
        };

        assert(is_domain_error(-std::numeric_limits<Real>::infinity()));
        for (Real x : sample_points<Real>())
          if (x > 0)
            assert(is_domain_error(-x));
      }
#endif // _LIBCPP_NO_EXCEPTIONS
  }

  { // compare against std::laguerre
    for (unsigned n = 0; n < g_max_n; ++n)
      for (Real x : sample_points<Real>())
        assert(std::assoc_laguerre(n, 0, x) == std::laguerre(n, x));
  }

  { // check against analytic polynoms for order n = 0..3, m = 0..127
    for (int m = 0; m < static_cast<int>(g_max_n); ++m) {
      for (Real x : sample_points<Real>()) {
        const Polynomial p0{Real{}, {1}, 1};
        const Polynomial p1{Real{}, {m + 1, -1}, 1};
        const Polynomial p2{Real{}, {(m + 1) * (m + 2), -2 * (m + 2), 1}, 2};
        const Polynomial p3{Real{}, {(m + 1) * (m + 2) * (m + 3), -3 * (m + 2) * (m + 3), 3 * (m + 3), -1}, 6};

        const CompareFloatingValues<Real> compare;
        assert(compare(std::assoc_laguerre(0, m, x), p0(x)));
        assert(compare(std::assoc_laguerre(1, m, x), p1(x)));
        assert(compare(std::assoc_laguerre(2, m, x), p2(x)));
        assert(compare(std::assoc_laguerre(3, m, x), p3(x)));
      }
    }
  }

  { // checks std::assoc_laguerref for bitwise equality with std::assoc_laguerre(unsigned, float)
    if constexpr (std::is_same_v<Real, float>)
      for (unsigned n = 0; n < g_max_n; ++n)
        for (unsigned m = 0; m < g_max_n; ++m)
          for (float x : sample_points<float>())
            assert(std::assoc_laguerre(n, m, x) == std::assoc_laguerref(n, m, x));
  }

  { // checks std::assoc_laguerrel for bitwise equality with std::assoc_laguerre(unsigned, long double)
    if constexpr (std::is_same_v<Real, long double>)
      for (unsigned n = 0; n < g_max_n; ++n)
        for (unsigned m = 0; m < g_max_n; ++m)
          for (long double x : sample_points<long double>())
            assert(std::assoc_laguerre(n, m, x) == std::assoc_laguerrel(n, m, x));
  }

#if 0
  { // evaluation at x=0: P_n(0) = 1
    const CompareFloatingValues<Real> compare;
    for (unsigned n = 0; n < g_max_n; ++n)
      assert(compare(std::assoc_laguerre(n, Real{0}), 1));
  }
#endif
}

struct TestFloat {
  template <class Real>
  static void operator()() {
    test<Real>();
  }
};

struct TestInt {
  template <class Integer>
  static void operator()() {
    // checks that std::assoc_laguerre(unsigned, Integer) actually wraps std::assoc_laguerre(unsigned, double)
    for (unsigned n = 0; n < g_max_n; ++n)
      for (unsigned m = 0; m < g_max_n; ++m)
        for (Integer x{0}; x < 20; ++x)
          assert(std::assoc_laguerre(n, m, x) == std::assoc_laguerre(n, m, static_cast<double>(x)));
  }
};

int main() {
  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::type_list<short, int, long, long long>(), TestInt());
}
