//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <cmath>

// double      assoc_legendre( unsigned l, unsigned m, double      x);
// float       assoc_legendre( unsigned l, unsigned m, float       x);
// long double assoc_legendre( unsigned l, unsigned m, long double x);
// float       assoc_legendref(unsigned l, unsigned m, float       x);
// long double assoc_legendrel(unsigned l, unsigned m, long double x);
// template <class Integer>
// double      assoc_legendre( unsigned l, unsigned m, Integer     x);

#include <array>
#include <cassert>
#include <functional>
#include <cmath>
#include <limits>
#include <iostream>
#include <numeric>

#include "type_algorithms.h"

inline constexpr unsigned g_max_lm = 128;

template <class T>
std::array<T, 7> sample_points() {
  return {-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0};
}

template <class Real>
class CompareFloatingValues {
private:
  Real tol;

public:
  CompareFloatingValues() {
    if (std::is_same_v<Real, float>)
      tol = 1e-6f;
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

template <class T, std::size_t N>
std::array<T, N> range() {
  std::array<T, N> arr;
  std::iota(arr.begin(), arr.end(), T{0});
  return arr;
}

template <class Real>
void test() {
#if 1
  { // checks if NaNs are reported correctly (i.e. output == input for input == NaN)
    using nl = std::numeric_limits<Real>;
    for (unsigned l = 0; l < g_max_lm; ++l)
      for (unsigned m = 0; m < g_max_lm; ++m)
        for (Real NaN : {nl::quiet_NaN(), nl::signaling_NaN()})
          assert(std::isnan(std::assoc_legendre(l, m, NaN)));
  }

  { // simple sample points for l,m=0..127 should not produce NaNs.
    for (unsigned l = 0; l < g_max_lm; ++l)
      for (unsigned m = 0; m < g_max_lm; ++m)
        for (Real x : sample_points<Real>())
          assert(!std::isnan(std::assoc_legendre(l, m, x)));
  }

  { // For x with abs(x) > 1 a domain_error exception should be thrown.
#  ifndef _LIBCPP_NO_EXCEPTIONS
    for (unsigned l = 0; l < g_max_lm; ++l)
      for (unsigned m = 0; m < g_max_lm; ++m) {
        const auto is_domain_error = [l, m](Real x) -> bool {
          try {
            std::assoc_legendre(l, m, x);
          } catch (const std::domain_error&) {
            return true;
          }
          return false;
        };

        Real inf = std::numeric_limits<Real>::infinity();
        for (Real absX : {std::nextafter(Real(1), inf), Real(2.0), Real(7.77), Real(42.42), inf})
          for (Real x : {-absX, absX})
            assert(is_domain_error(x));
      }
#  endif // _LIBCPP_NO_EXCEPTIONS
  }
#endif

#if 1
  { // check against analytic polynoms for order n=0..6
    using Func     = std::function<Real(Real)>;
    const Func P00 = [](Real) { return 1; };

    const Func P10 = [](Real y) -> Real { return y; };
    const Func P11 = [](Real y) -> Real { return std::sqrt((1 - y) * (1 + y)); };

    const Func P20 = [](Real y) -> Real { return (3 * y * y - 1) / 2; };
    const Func P21 = [](Real y) -> Real { return 3 * y * std::sqrt((1 - y) * (1 + y)); };
    const Func P22 = [](Real y) -> Real { return 3 * (1 - y) * (1 + y); };

    const Func P30 = [](Real y) -> Real { return (5 * y * y - 3) * y / 2; };
    const Func P31 = [](Real y) -> Real { return Real(1.5) * (5 * y * y - 1) * std::sqrt((1 - y) * (1 + y)); };
    const Func P32 = [](Real y) -> Real { return 15 * y * (1 - y) * (1 + y); };
    const Func P33 = [](Real y) -> Real { return 15 * std::pow((1 - y) * (1 + y), Real(1.5)); };

    const Func P40 = [](Real y) -> Real { return (35 * std::pow(y, 4) - 30 * y * y + 3) / 8; };
    const Func P41 = [](Real y) -> Real { return Real(2.5) * y * (7 * y * y - 3) * std::sqrt((1 - y) * (1 + y)); };
    const Func P42 = [](Real y) -> Real { return Real(7.5) * (7 * y * y - 1) * (1 - y) * (1 + y); };
    const Func P43 = [](Real y) -> Real { return 105 * y * std::pow((1 - y) * (1 + y), Real(1.5)); };
    const Func P44 = [](Real y) -> Real { return 105 * std::pow((1 - y) * (1 + y), Real(2)); };

    unsigned l = 0;
    unsigned m = 0;
    for (auto P : {P00, P10, P11, P20, P21, P22, P30, P31, P32, P33, P40, P41, P42, P43, P44}) {
      for (Real x : sample_points<Real>()) {
        const CompareFloatingValues<Real> compare;
        assert(compare(std::assoc_legendre(l, m, x), P(x)));
      }

      if (l == m) {
        ++l;
        m = 0;
      } else
        ++m;
    }
  }
#endif

#if 1
  { // checks std::assoc_legendref for bitwise equality with std::assoc_legendre(unsigned, float)
    if constexpr (std::is_same_v<Real, float>)
      for (unsigned l = 0; l < g_max_lm; ++l)
        for (unsigned m = 0; m < g_max_lm; ++m)
          for (float x : sample_points<float>())
            assert(std::assoc_legendre(l, m, x) == std::assoc_legendref(l, m, x));
  }

  { // checks std::assoc_legendrel for bitwise equality with std::assoc_legendre(unsigned, long double)
    if constexpr (std::is_same_v<Real, long double>)
      for (unsigned l = 0; l < g_max_lm; ++l)
        for (unsigned m = 0; m < g_max_lm; ++m)
          for (long double x : sample_points<long double>())
            assert(std::assoc_legendre(l, m, x) == std::assoc_legendrel(l, m, x));
  }
#endif

#if 0
  { // evaluation at x=1: P_n(1) = 1
    const CompareFloatingValues<Real> compare;
    for (unsigned n = 0; n < g_max_lm; ++n)
      assert(compare(std::assoc_legendre(l, m, Real{1}), 1));
  }

  { // evaluation at x=-1: P_n(-1) = (-1)^n
    const CompareFloatingValues<Real> compare;
    for (unsigned n = 0; n < g_max_lm; ++n)
      assert(compare(std::assoc_legendre(l, m, Real{-1}), std::pow(-1, n)));
  }

  { // evaluation at x=0:
    //    P_{2n  }(0) = (-1)^n (2n-1)!! / (2n)!!
    //    P_{2n+1}(0) = 0
    const CompareFloatingValues<Real> compare;
    Real doubleFactorials{1};
    for (unsigned n = 0; n < g_max_lm; ++n) {
      if (n & 1) // odd
        doubleFactorials *= n;
      else if (n != 0) // even and not zero
        doubleFactorials /= n;

      assert(compare(std::assoc_legendre(l, m, Real{0}), (n & 1) ? Real{0} : std::pow(-1, n / 2) * doubleFactorials));
    }
  }
#endif
}

struct TestFloat {
  template <class Real>
  void operator()() {
    test<Real>();
  }
};

struct TestInt {
  template <class Integer>
  void operator()() {
    // checks that std::assoc_legendre(unsigned, unsigned, Integer) actually wraps
    // std::assoc_legendre(unsigned, unsigned, double)
    for (unsigned l = 0; l < g_max_lm; ++l)
      for (unsigned m = 0; m < g_max_lm; ++m)
        for (Integer x : {-1, 0, 1})
          assert(std::assoc_legendre(l, m, x) == std::assoc_legendre(l, m, static_cast<double>(x)));
  }
};

int main() {
  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::type_list<short, int, long, long long>(), TestInt());
}
