//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <cmath>

// double         laguerre(unsigned n, double x);
// float          laguerre(unsigned n, float x);
// long double    laguerre(unsigned n, long double x);
// float          laguerref(unsigned n, float x);
// long double    laguerrel(unsigned n, long double x);
// template <class Integer>
// double         laguerre(unsigned n, Integer x);

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
    for (Real NaN : {nl::quiet_NaN(), nl::signaling_NaN()})
      for (unsigned n = 0; n < g_max_n; ++n)
        assert(std::isnan(std::laguerre(n, NaN)));
  }

  { // simple sample points for n=0..127 should not produce NaNs.
    for (Real x : sample_points<Real>())
      for (unsigned n = 0; n < g_max_n; ++n)
        assert(!std::isnan(std::laguerre(n, x)));
  }

  { // For any $x < 0$ a domain_error exception should be thrown.
#ifndef _LIBCPP_NO_EXCEPTIONS
    for (unsigned n = 0; n < g_max_n; ++n) {
      const auto is_domain_error = [n](Real x) -> bool {
        try {
          std::laguerre(n, x);
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

  { // check against analytic polynoms for order n=0..6
    for (Real x : sample_points<Real>()) {
      const Polynomial p0{Real{}, {1}, 1};
      const Polynomial p1{Real{}, {1, -1}, 1};
      const Polynomial p2{Real{}, {2, -4, 1}, 2};
      const Polynomial p3{Real{}, {6, -18, 9, -1}, 6};
      const Polynomial p4{Real{}, {24, -96, 72, -16, 1}, 24};
      const Polynomial p5{Real{}, {120, -600, 600, -200, 25, -1}, 120};
      const Polynomial p6{Real{}, {720, -4320, 5400, -2400, 450, -36, 1}, 720};

      const CompareFloatingValues<Real> compare;
      assert(compare(std::laguerre(0, x), p0(x)));
      assert(compare(std::laguerre(1, x), p1(x)));
      assert(compare(std::laguerre(2, x), p2(x)));
      assert(compare(std::laguerre(3, x), p3(x)));
      assert(compare(std::laguerre(4, x), p4(x)));
      assert(compare(std::laguerre(5, x), p5(x)));
      assert(compare(std::laguerre(6, x), p6(x)));
    }
  }

  { // checks std::laguerref for bitwise equality with std::laguerre(unsigned, float)
    if constexpr (std::is_same_v<Real, float>)
      for (unsigned n = 0; n < g_max_n; ++n)
        for (float x : sample_points<float>())
          assert(std::laguerre(n, x) == std::laguerref(n, x));
  }

  { // checks std::laguerrel for bitwise equality with std::laguerre(unsigned, long double)
    if constexpr (std::is_same_v<Real, long double>)
      for (unsigned n = 0; n < g_max_n; ++n)
        for (long double x : sample_points<long double>())
          assert(std::laguerre(n, x) == std::laguerrel(n, x));
  }

  { // evaluation at x=0: P_n(0) = 1
    const CompareFloatingValues<Real> compare;
    for (unsigned n = 0; n < g_max_n; ++n)
      assert(compare(std::laguerre(n, Real{0}), 1));
  }

  { // evaluation at x=+inf
    Real inf = std::numeric_limits<Real>::infinity();
    for (unsigned n = 1; n < g_max_n; ++n)
      assert(std::laguerre(n, inf) == ((n & 1) ? -inf : inf));
  }

  { // check: if overflow occurs that it is mapped to the correct infinity
    constexpr auto check_for_overflow = [](unsigned n_threshold, Real x) {
      for (unsigned n = 0; n < g_max_n; ++n) {
        if (n < n_threshold)
          assert(std::isfinite(std::laguerre(n, x)));
        else {
          Real inf = std::numeric_limits<Real>::infinity();

          // alternating limits (+-inf) only holds for x > largest root which is ~480.5 for order n=127.
          assert(x > 481);
          assert(std::laguerre(n, x) == ((n & 1) ? -inf : inf));
        }
      }
    };

    if constexpr (std::is_same_v<Real, float>) {
      static_assert(sizeof(float) == 4);
      check_for_overflow(23, 500.0f);
    } else if constexpr (std::is_same_v<Real, double>) {
      static_assert(sizeof(double) == 8);
      check_for_overflow(116, 20'000.0);
    } else if constexpr (std::is_same_v<Real, long double>) {
      static_assert(sizeof(long double) == 16);
      check_for_overflow(50, 1e100l);
    }
  }
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
    // checks that std::laguerre(unsigned, Integer) actually wraps std::laguerre(unsigned, double)
    for (unsigned n = 0; n < g_max_n; ++n)
      for (Integer x{0}; x < 20; ++x)
        assert(std::laguerre(n, x) == std::laguerre(n, static_cast<double>(x)));
  }
};

int main() {
  types::for_each(types::floating_point_types(), TestFloat());
  // types::for_each(types::type_list<short, int, long, long long>(), TestInt());
}
