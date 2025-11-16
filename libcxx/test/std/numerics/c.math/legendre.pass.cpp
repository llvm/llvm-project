//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <cmath>

// double         legendre(unsigned n, double x);
// float          legendre(unsigned n, float x);
// long double    legendre(unsigned n, long double x);
// float          legendref(unsigned n, float x);
// long double    legendrel(unsigned n, long double x);
// template <class Integer>
// double         legendre(unsigned n, Integer x);

#include <array>
#include <cassert>
#include <cmath>
#include <limits>

#include "type_algorithms.h"

inline constexpr unsigned g_max_n = 128;

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

template <class Real>
void test() {
  { // checks if NaNs are reported correctly (i.e. output == input for input == NaN)
    using nl = std::numeric_limits<Real>;
    for (Real NaN : {nl::quiet_NaN(), nl::signaling_NaN()})
      for (unsigned n = 0; n < g_max_n; ++n)
        assert(std::isnan(std::legendre(n, NaN)));
  }

  { // simple sample points for n=0..127 should not produce NaNs.
    for (Real x : sample_points<Real>())
      for (unsigned n = 0; n < g_max_n; ++n)
        assert(!std::isnan(std::legendre(n, x)));
  }

  { // For x with abs(x) > 1 an domain_error exception should be thrown.
#ifndef _LIBCPP_NO_EXCEPTIONS
    for (double absX : {2.0, 7.77, 42.42, std::numeric_limits<double>::infinity()})
      for (Real x : {-absX, absX})
        for (unsigned n = 0; n < g_max_n; ++n) {
          bool throws = false;
          try {
            std::legendre(n, x);
          } catch (const std::domain_error&) {
            throws = true;
          }
          assert(throws);
        }
#endif // _LIBCPP_NO_EXCEPTIONS
  }

  { // check against analytic polynoms for order n=0..6
    for (Real x : sample_points<Real>()) {
      const auto l0 = [](Real) -> Real { return 1; };
      const auto l1 = [](Real y) -> Real { return y; };
      const auto l2 = [](Real y) -> Real { return (3 * y * y - 1) / 2; };
      const auto l3 = [](Real y) -> Real { return (5 * y * y - 3) * y / 2; };
      const auto l4 = [](Real y) -> Real { return (35 * std::pow(y, 4) - 30 * y * y + 3) / 8; };
      const auto l5 = [](Real y) -> Real { return (63 * std::pow(y, 4) - 70 * y * y + 15) * y / 8; };
      const auto l6 = [](Real y) -> Real {
        return (231 * std::pow(y, 6) - 315 * std::pow(y, 4) + 105 * y * y - 5) / 16;
      };

      const CompareFloatingValues<Real> compare;
      assert(compare(std::legendre(0, x), l0(x)));
      assert(compare(std::legendre(1, x), l1(x)));
      assert(compare(std::legendre(2, x), l2(x)));
      assert(compare(std::legendre(3, x), l3(x)));
      assert(compare(std::legendre(4, x), l4(x)));
      assert(compare(std::legendre(5, x), l5(x)));
      assert(compare(std::legendre(6, x), l6(x)));
    }
  }

  { // checks std::legendref for bitwise equality with std::legendre(unsigned, float)
    if constexpr (std::is_same_v<Real, float>)
      for (unsigned n = 0; n < g_max_n; ++n)
        for (float x : sample_points<float>())
          assert(std::legendre(n, x) == std::legendref(n, x));
  }

  { // checks std::legendrel for bitwise equality with std::legendre(unsigned, long double)
    if constexpr (std::is_same_v<Real, long double>)
      for (unsigned n = 0; n < g_max_n; ++n)
        for (long double x : sample_points<long double>()) {
          assert(std::legendre(n, x) == std::legendrel(n, x));
          assert(std::legendre(n, x) == std::legendrel(n, x));
        }
  }

  { // evaluation at x=1: P_n(1) = 1
    const CompareFloatingValues<Real> compare;
    for (unsigned n = 0; n < g_max_n; ++n)
      assert(compare(std::legendre(n, Real{1}), 1));
  }

  { // evaluation at x=-1: P_n(-1) = (-1)^n
    const CompareFloatingValues<Real> compare;
    for (unsigned n = 0; n < g_max_n; ++n)
      assert(compare(std::legendre(n, Real{-1}), std::pow(-1, n)));
  }

  { // evaluation at x=0:
    //    P_{2n  }(0) = (-1)^n (2n-1)!! / (2n)!!
    //    P_{2n+1}(0) = 0
    const CompareFloatingValues<Real> compare;
    Real doubleFactorials{1};
    for (unsigned n = 0; n < g_max_n; ++n) {
      if (n & 1) // odd
        doubleFactorials *= n;
      else if (n != 0) // even and not zero
        doubleFactorials /= n;

      assert(compare(std::legendre(n, Real{0}), (n & 1) ? Real{0} : std::pow(-1, n / 2) * doubleFactorials));
    }
  }
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
    // checks that std::legendre(unsigned, Integer) actually wraps std::legendre(unsigned, double)
    for (unsigned n = 0; n < g_max_n; ++n)
      for (Integer x : {-1, 0, 1})
        assert(std::legendre(n, x) == std::legendre(n, static_cast<double>(x)));
  }
};

int main() {
  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::type_list<short, int, long, long long>(), TestInt());
}
