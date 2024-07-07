//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <experimental/cmath>

#include <cassert>
#include <experimental/math>
#include <limits>

#if _LIBCPP_STD_VER > 14

template <class T> void testLegendreNaNPropagation() {
  const unsigned MaxN = 127;
  const T x = std::numeric_limits<T>::quiet_NaN();
  for (unsigned n = 0; n <= MaxN; ++n) {
    assert(std::isnan(std::experimental::legendre(n, x)));
  }
}

template <class T> void testLegendreNotNaN(const T x) {
  assert(!std::isnan(x));
  const unsigned MaxN = 127;
  for (unsigned n = 0; n <= MaxN; ++n) {
    assert(!std::isnan(std::experimental::legendre(n, x)));
  }
}

template <class T> void testLegendreThrows(const T x) {
#ifndef _LIBCPP_NO_EXCEPTIONS
  const unsigned MaxN = 127;
  for (unsigned n = 0; n <= MaxN; ++n) {
    bool Throws = false;
    try {
      std::experimental::legendre(n, x);
    } catch (const std::domain_error &) {
      Throws = true;
    }
    assert(Throws);
  }
#endif // _LIBCPP_NO_EXCEPTIONS
}

template <class T>
void testLegendreAnalytic(const T x, const T AbsTolerance,
                          const T RelTolerance) {
  assert(!std::isnan(x));
  const auto compareFloatingPoint =
      [AbsTolerance, RelTolerance](const T Result, const T ExpectedResult) {
        if (std::isinf(ExpectedResult) && std::isinf(Result))
          return true;

        if (std::isnan(ExpectedResult) || std::isnan(Result))
          return false;

        const T Tolerance =
            AbsTolerance + std::abs(ExpectedResult) * RelTolerance;
        return std::abs(Result - ExpectedResult) < Tolerance;
      };

  const auto l0 = [](T) { return T(1); };
  const auto l1 = [](T y) { return y; };
  const auto l2 = [](T y) { return (T(3) * y * y - T(1)) / T(2); };
  const auto l3 = [](T y) { return (T(5) * y * y - T(3)) * y / T(2); };
  const auto l4 = [](T y) {
    return (T(35) * y * y * y * y - T(30) * y * y + T(3)) / T(8);
  };
  const auto l5 = [](T y) {
    return (T(63) * y * y * y * y - T(70) * y * y + T(15)) * y / T(8);
  };
  const auto l6 = [](T y) {
    const T y2 = y * y;
    return (T(231) * y2 * y2 * y2 - T(315) * y2 * y2 + T(105) * y2 - T(5)) /
           T(16);
  };

  assert(compareFloatingPoint(std::experimental::legendre(0, x), l0(x)));
  assert(compareFloatingPoint(std::experimental::legendre(1, x), l1(x)));
  assert(compareFloatingPoint(std::experimental::legendre(2, x), l2(x)));
  assert(compareFloatingPoint(std::experimental::legendre(3, x), l3(x)));
  assert(compareFloatingPoint(std::experimental::legendre(4, x), l4(x)));
  assert(compareFloatingPoint(std::experimental::legendre(5, x), l5(x)));
  assert(compareFloatingPoint(std::experimental::legendre(6, x), l6(x)));
}

template <class T>
void testLegendre(const T AbsTolerance, const T RelTolerance) {
  testLegendreNaNPropagation<T>();
  testLegendreThrows<T>(T(-5));
  testLegendreThrows<T>(T(5));

  const T Samples[] = {T(-1.0), T(-0.5), T(-0.1), T(0.0),
                       T(0.1),  T(0.5),  T(1.0)};

  for (T x : Samples) {
    testLegendreNotNaN(x);
    testLegendreAnalytic(x, AbsTolerance, RelTolerance);
  }
}

#endif

int main(int, char **) {
#if _LIBCPP_STD_VER > 14
  testLegendre<float>(1e-6f, 1e-6f);
  testLegendre<double>(1e-9, 1e-9);
  testLegendre<long double>(1e-12, 1e-12);
#endif
  return 0;
}
