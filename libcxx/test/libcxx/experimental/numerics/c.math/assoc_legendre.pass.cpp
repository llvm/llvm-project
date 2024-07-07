//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <experimental/cmath>

#include <cassert>
#include <experimental/cmath>
#include <limits>

#if _LIBCPP_STD_VER > 14

template <class T> void testAssocLegendreNaNPropagation() {
  const unsigned MaxN = 127;
  const T x = std::numeric_limits<T>::quiet_NaN();
  for (unsigned n = 0; n <= MaxN; ++n) {
    for (unsigned m = 0; m <= MaxN; ++m) {
      assert(std::isnan(std::experimental::assoc_legendre(n, m, x)));
    }
  }
}

template <class T> void testAssocLegendreNotNaN(const T x) {
  assert(!std::isnan(x));
  const unsigned MaxN = 127;
  for (unsigned n = 0; n <= MaxN; ++n) {
    for (unsigned m = 0; m <= MaxN; ++m) {
      assert(!std::isnan(std::experimental::assoc_legendre(n, m, x)));
    }
  }
}

template <class T> void testAssocLegendreThrows(const T x) {
#ifndef _LIBCPP_NO_EXCEPTIONS
  const unsigned MaxN = 127;
  for (unsigned n = 0; n <= MaxN; ++n) {
    for (unsigned m = 0; m <= MaxN; ++m) {
      bool Throws = false;
      try {
        std::experimental::assoc_legendre(n, m, x);
      } catch (const std::domain_error &) {
        Throws = true;
      }
      assert(Throws);
    }
  }
#endif // _LIBCPP_NO_EXCEPTIONS
}

template <class T>
void testAssocLegendreVsLegendre(const T x, const T AbsTolerance,
                                 const T RelTolerance) {
  assert(!std::isnan(x));
  const unsigned MaxN = 127;
  for (unsigned n = 0; n <= MaxN; ++n) {
    for (unsigned m = 0; m <= MaxN; ++m) {
      const T Result = std::experimental::assoc_legendre(n, 0, x);
      const T ExpectedResult = std::experimental::legendre(n, x);
      const T Tolerance =
          AbsTolerance + std::abs(ExpectedResult) * RelTolerance;
      const T Difference = std::abs(Result - ExpectedResult);
      assert(Difference <= Tolerance);
    }
  }
}

template <class T>
void testAssocLegendreAnalytic(const T x, const T AbsTolerance,
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

  const auto l00 = [](T) { return T(1); };

  const auto l10 = [](T x) { return x; };
  const auto l11 = [](T x) { return std::sqrt((T(1) - x) * (T(1) + x)); };

  const auto l20 = [](T x) { return (T(3) * x * x - T(1)) / T(2); };
  const auto l21 = [](T x) {
    return T(3) * x * std::sqrt((T(1) - x) * (T(1) + x));
  };
  const auto l22 = [](T x) { return T(3) * (T(1) - x) * (T(1) + x); };

  const auto l30 = [](T x) { return (T(5) * x * x - T(3)) * x / T(2); };
  const auto l31 = [](T x) {
    return T(3) / T(2) * (T(5) * x * x - T(1)) *
           std::sqrt((T(1) - x) * (T(1) + x));
  };
  const auto l32 = [](T x) { return T(15) * x * (T(1) - x) * (T(1) + x); };
  const auto l33 = [](T x) {
    const T temp = (T(1) - x) * (T(1) + x);
    return T(15) * temp * std::sqrt(temp);
  };

  const auto l40 = [](T x) {
    return (T(35) * x * x * x * x - T(30) * x * x + T(3)) / T(8);
  };
  const auto l41 = [](T x) {
    return T(5) / T(2) * x * (T(7) * x * x - T(3)) *
           std::sqrt((T(1) - x) * (T(1) + x));
  };
  const auto l42 = [](T x) {
    return T(15) / T(2) * (T(7) * x * x - 1) * (T(1) - x) * (T(1) + x);
  };
  const auto l43 = [](T x) {
    const T temp = (T(1) - x) * (T(1) + x);
    return T(105) * x * temp * std::sqrt(temp);
  };
  const auto l44 = [](T x) {
    const T temp = (T(1) - x) * (T(1) + x);
    return T(105) * temp * temp;
  };

  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(0, 0, x), l00(x)));

  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(1, 0, x), l10(x)));
  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(1, 1, x), l11(x)));

  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(2, 0, x), l20(x)));
  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(2, 1, x), l21(x)));
  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(2, 2, x), l22(x)));

  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(3, 0, x), l30(x)));
  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(3, 1, x), l31(x)));
  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(3, 2, x), l32(x)));
  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(3, 3, x), l33(x)));

  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(4, 0, x), l40(x)));
  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(4, 1, x), l41(x)));
  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(4, 2, x), l42(x)));
  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(4, 3, x), l43(x)));
  assert(
      compareFloatingPoint(std::experimental::assoc_legendre(4, 4, x), l44(x)));

  try {
    const unsigned MaxN = 127;
    for (unsigned n = 0; n <= MaxN; ++n) {
      for (unsigned m = n + 1; m <= MaxN; ++m) {
        assert(std::experimental::assoc_legendre(n, m, x) <= AbsTolerance);
      }
    }
  } catch (const std::domain_error &) {
    // Should not throw! The expression given in
    // ISO/IEC JTC 1/SC 22/WG 21 N3060 is actually well-defined for m > n!
    assert(false);
  }
}

template <class T>
void testAssocLegendre(const T AbsTolerance, const T RelTolerance) {
  testAssocLegendreNaNPropagation<T>();
  testAssocLegendreThrows<T>(T(-5));
  testAssocLegendreThrows<T>(T(5));

  const T Samples[] = {T(-1.0), T(-0.5), T(-0.1), T(0.0),
                       T(0.1),  T(0.5),  T(1.0)};

  for (T x : Samples) {
    testAssocLegendreNotNaN(x);
    testAssocLegendreVsLegendre(x, AbsTolerance, RelTolerance);
    testAssocLegendreAnalytic(x, AbsTolerance, RelTolerance);
  }
}

#endif

int main(int, char **) {
#if _LIBCPP_STD_VER > 14
  testAssocLegendre<float>(1e-6f, 1e-6f);
  testAssocLegendre<double>(1e-9, 1e-9);
  testAssocLegendre<long double>(1e-12, 1e-12);
#endif
  return 0;
}
