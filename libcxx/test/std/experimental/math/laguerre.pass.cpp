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

template <class T> void testLaguerreNaNPropagation() {
  const unsigned MaxN = 127;
  const T x = std::numeric_limits<T>::quiet_NaN();
  for (unsigned n = 0; n <= MaxN; ++n) {
    assert(std::isnan(std::experimental::laguerre(n, x)));
  }
}

template <class T> void testLaguerreNotNaN(const T x) {
  assert(!std::isnan(x));
  const unsigned MaxN = 127;
  for (unsigned n = 0; n <= MaxN; ++n) {
    assert(!std::isnan(std::experimental::laguerre(n, x)));
  }
}

template <class T> void testLaguerreThrows(const T x) {
#ifndef _LIBCPP_NO_EXCEPTIONS
  const unsigned MaxN = 127;
  for (unsigned n = 0; n <= MaxN; ++n) {
    bool Throws = false;
    try {
      std::experimental::laguerre(n, x);
    } catch (const std::domain_error &) {
      Throws = true;
    }
    assert(Throws);
  }
#endif // _LIBCPP_NO_EXCEPTIONS
}

template <class T>
void testLaguerreAnalytic(const T x, const T AbsTolerance,
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
  const auto l1 = [](T y) { return -y + 1; };
  const auto l2 = [](T y) { return (y * y - T(4) * y + T(2)) / T(2); };
  const auto l3 = [](T y) {
    return (-y * y * y + T(9) * y * y - T(18) * y + T(6)) / T(6);
  };
  const auto l4 = [](T y) {
    return (y * y * y * y - T(16) * y * y * y + T(72) * y * y - T(96) * y +
            T(24)) /
           T(24);
  };
  const auto l5 = [](T y) {
    return (-y * y * y * y * y + T(25) * y * y * y * y - T(200) * y * y * y +
            T(600) * y * y - T(600) * y + T(120)) /
           T(120);
  };
  const auto l6 = [](T y) {
    return (y * y * y * y * y * y - T(36) * y * y * y * y * y +
            T(450) * y * y * y * y - T(2400) * y * y * y + T(5400) * y * y -
            T(4320) * y + T(720)) /
           T(720);
  };

  assert(compareFloatingPoint(std::experimental::laguerre(0, x), l0(x)));
  assert(compareFloatingPoint(std::experimental::laguerre(1, x), l1(x)));
  assert(compareFloatingPoint(std::experimental::laguerre(2, x), l2(x)));
  assert(compareFloatingPoint(std::experimental::laguerre(3, x), l3(x)));
  assert(compareFloatingPoint(std::experimental::laguerre(4, x), l4(x)));
  assert(compareFloatingPoint(std::experimental::laguerre(5, x), l5(x)));
  assert(compareFloatingPoint(std::experimental::laguerre(6, x), l6(x)));
}

template <class T>
void testLaguerre(const T AbsTolerance, const T RelTolerance) {
  testLaguerreNaNPropagation<T>();
  testLaguerreThrows<T>(T(-5));

  const T Samples[] = {T(0.0), T(0.1), T(0.5), T(1.0), T(10.0)};

  for (T x : Samples) {
    testLaguerreNotNaN(x);
    testLaguerreAnalytic(x, AbsTolerance, RelTolerance);
  }
}

#endif

int main(int, char **) {
#if _LIBCPP_STD_VER > 14
  testLaguerre<float>(1e-6f, 1e-6f);
  testLaguerre<double>(1e-9, 1e-9);
  testLaguerre<long double>(1e-12, 1e-12);
#endif
  return 0;
}
