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

template <class T> void testAssocLaguerreNaNPropagation() {
  const unsigned MaxN = 127;
  const T x = std::numeric_limits<T>::quiet_NaN();
  for (unsigned n = 0; n <= MaxN; ++n) {
    for (unsigned m = 0; m <= MaxN; ++m) {
      assert(std::isnan(std::experimental::assoc_laguerre(n, m, x)));
    }
  }
}

template <class T> void testAssocLaguerreNotNaN(const T x) {
  assert(!std::isnan(x));
  const unsigned MaxN = 127;
  for (unsigned n = 0; n <= MaxN; ++n) {
    for (unsigned m = 0; m <= MaxN; ++m) {
      assert(!std::isnan(std::experimental::assoc_laguerre(n, m, x)));
    }
  }
}

template <class T> void testAssocLaguerreThrows(const T x) {
#ifndef _LIBCPP_NO_EXCEPTIONS
  const unsigned MaxN = 127;
  for (unsigned n = 0; n <= MaxN; ++n) {
    for (unsigned m = 0; m <= MaxN; ++m) {
      bool Throws = false;
      try {
        std::experimental::assoc_laguerre(n, m, x);
      } catch (const std::domain_error &) {
        Throws = true;
      }
      assert(Throws);
    }
  }
#endif // _LIBCPP_NO_EXCEPTIONS
}

template <class T>
void testAssocLaguerreVsLaguerre(const T x, const T AbsTolerance,
                                 const T RelTolerance) {
  assert(!std::isnan(x));
  const unsigned MaxN = 127;
  for (unsigned n = 0; n <= MaxN; ++n) {
    for (unsigned m = 0; m <= MaxN; ++m) {
      const T Result = std::experimental::assoc_laguerre(n, 0, x);
      const T ExpectedResult = std::experimental::laguerre(n, x);
      const T Tolerance =
          AbsTolerance + std::abs(ExpectedResult) * RelTolerance;
      const T Difference = std::abs(Result - ExpectedResult);
      assert(Difference <= Tolerance);
    }
  }
}

template <class T>
void testAssocLaguerreAnalytic(const T x, const T AbsTolerance,
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

  const auto l0 = [](T, unsigned) { return T(1); };
  const auto l1 = [](T x, unsigned m) { return -x + T(m + 1); };
  const auto l2 = [](T x, unsigned m) {
    return x * x / T(2) - T(m + 2) * x + T(m + 1) * T(m + 2) / T(2);
  };
  const auto l3 = [](T x, unsigned m) {
    return -x * x * x / T(6) + T(m + 3) * x * x / T(2) -
           T(m + 2) * T(m + 3) * x / T(2) +
           T(m + 1) * T(m + 2) * T(m + 3) / T(6);
  };

  for (unsigned m = 0; m < 128; ++m) {
    assert(compareFloatingPoint(std::experimental::assoc_laguerre(0, m, x),
                                l0(x, m)));
    assert(compareFloatingPoint(std::experimental::assoc_laguerre(1, m, x),
                                l1(x, m)));
    assert(compareFloatingPoint(std::experimental::assoc_laguerre(2, m, x),
                                l2(x, m)));
    assert(compareFloatingPoint(std::experimental::assoc_laguerre(3, m, x),
                                l3(x, m)));
  }
}

template <class T>
void testAssocLaguerre(const T AbsTolerance, const T RelTolerance) {
  testAssocLaguerreNaNPropagation<T>();
  testAssocLaguerreThrows<T>(T(-5));

  const T Samples[] = {T(0.0), T(0.1), T(0.5), T(1.0), T(10.0)};

  for (T x : Samples) {
    testAssocLaguerreNotNaN(x);
    testAssocLaguerreAnalytic(x, AbsTolerance, RelTolerance);
    testAssocLaguerreVsLaguerre(x, AbsTolerance, RelTolerance);
  }
}

#endif

int main(int, char **) {
#if _LIBCPP_STD_VER > 14
  testAssocLaguerre<float>(1e-5f, 1e-5f);
  testAssocLaguerre<double>(1e-9, 1e-9);
  testAssocLaguerre<long double>(1e-12, 1e-12);
#endif
  return 0;
}
