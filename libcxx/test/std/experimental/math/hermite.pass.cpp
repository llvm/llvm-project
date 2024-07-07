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
#include <iostream>
#include <limits>
#include <vector>

#if _LIBCPP_STD_VER > 14

template <class T> void testHermiteNaNPropagation() {
  const unsigned MaxN = 127;
  const T x = std::numeric_limits<T>::quiet_NaN();
  for (unsigned n = 0; n <= MaxN; ++n) {
    assert(std::isnan(std::experimental::hermite(n, x)));
  }
}

template <class T> void testHermiteNotNaN(const T x) {
  assert(!std::isnan(x));
  const unsigned MaxN = 127;
  for (unsigned n = 0; n <= MaxN; ++n) {
    assert(!std::isnan(std::experimental::hermite(n, x)));
  }
}

template <class T>
void testHermiteAnalytic(const T x, const T AbsTolerance,
                         const T RelTolerance) {
  assert(!std::isnan(x));
  const auto compareFloatingPoint =
      [AbsTolerance, RelTolerance](const T Result, const T Expected) {
        if (std::isinf(Expected) && std::isinf(Result))
          return true;

        if (std::isnan(Expected) || std::isnan(Result))
          return false;

        const T Tolerance = AbsTolerance + std::abs(Expected) * RelTolerance;
        return std::abs(Result - Expected) < Tolerance;
      };

  const auto h0 = [](T) { return T(1); };
  const auto h1 = [](T y) { return T(2) * y; };
  const auto h2 = [](T y) { return T(4) * y * y - T(2); };
  const auto h3 = [](T y) { return y * (T(8) * y * y - T(12)); };
  const auto h4 = [](T y) {
    return (T(16) * y * y * y * y - T(48) * y * y + T(12));
  };
  const auto h5 = [](T y) {
    return y * (T(32) * y * y * y * y - T(160) * y * y + T(120));
  };

  assert(compareFloatingPoint(std::experimental::hermite(0, x), h0(x)));
  assert(compareFloatingPoint(std::experimental::hermite(1, x), h1(x)));
  assert(compareFloatingPoint(std::experimental::hermite(2, x), h2(x)));
  assert(compareFloatingPoint(std::experimental::hermite(3, x), h3(x)));
  assert(compareFloatingPoint(std::experimental::hermite(4, x), h4(x)));
  assert(compareFloatingPoint(std::experimental::hermite(5, x), h5(x)));
}

/// \details This method checks if the following recurrence relation holds:
/// \f[
/// H_{n+1}(x) = 2x H_{n}(x) - 2n H_{n-1}(x)
/// \f]
template <class T>
void testRecurrenceRelation(T x, T RelTolerance, T AbsTolerance) {
  const unsigned MaxN = 127;
  for (unsigned n = 1; n < MaxN; ++n) {
    const T HermiteNext = std::experimental::hermite(n + 1, x);
    const T HermiteNextRecurrence =
        T(2) * x * std::experimental::hermite(n, x) -
        T(2) * T(n) * std::experimental::hermite(n - 1, x);
    const T Tolerance = AbsTolerance + std::abs(HermiteNext) * RelTolerance;
    const T Error = std::abs(HermiteNextRecurrence - HermiteNext);

    if (std::isinf(HermiteNext))
      break;
    assert(Error < Tolerance);
  }
}

template <class T> void testRecurrenceRelation(T RelTolerance, T AbsTolerance) {
  const T Samples[] = {T(-1.0), T(-0.5), T(-0.1), T(0.0),
                       T(0.1),  T(0.5),  T(1.0)};
  for (T x : Samples)
    testRecurrenceRelation(x, RelTolerance, AbsTolerance);
}

/// \note Roots are taken from
/// Salzer, Herbert E., Ruth Zucker, and Ruth Capuano.
/// Table of the zeros and weight factors of the first twenty Hermite
/// polynomials. US Government Printing Office, 1952.
template <class T> std::vector<T> getHermiteRoots(unsigned n) {
  if (n == 0u)
    return {};
  if (n == 1u)
    return {T(0)};
  if (n == 2u)
    return {T(0.707106781186548)};
  if (n == 3u)
    return {T(0),
            T(1.224744871391589)};
  if (n == 4u)
    return {T(0.524647623275290),
            T(1.650680123885785)};
  if (n == 5u)
    return {T(0), T(0.958572464613819),
            T(2.020182870456086)};
  if (n == 6u)
    return {T(0.436077411927617),
            T(1.335849074013697),
            T(2.350604973674492)};
  if (n == 7u)
    return {T(0),
            T(0.816287882858965),
            T(1.673551628767471),
            T(2.651961356835233)};
  if (n == 8u)
    return {T(0.381186990207322),
            T(1.157193712446780),
            T(1.981656756695843),
            T(2.930637420257244)};
  if (n == 9u)
    return {T(0),
            T(0.723551018752838),
            T(1.468553289216668),
            T(2.266580584531843),
            T(3.190993201781528)};
  if (n == 10u)
    return {T(0.342901327223705),
            T(1.036610829789514),
            T(1.756683649299882),
            T(2.532731674232790),
            T(3.436159118837738)};
  if (n == 11u)
    return {T(0),
            T(0.65680956682100),
            T(1.326557084494933),
            T(2.025948015825755),
            T(2.783290099781652),
            T(3.668470846559583)};

  if (n == 12u)
    return {T(0.314240376254359),
            T(0.947788391240164),
            T(1.597682635152605),
            T(2.279507080501060),
            T(3.020637025120890),
            T(3.889724897869782)};

  if (n == 13u)
    return {T(0),
            T(0.605763879171060),
            T(1.220055036590748),
            T(1.853107651601512),
            T(2.519735685678238),
            T(3.246608978372410),
            T(4.101337596178640)};

  if (n == 14u)
    return {T(0.29174551067256),
            T(0.87871378732940),
            T(1.47668273114114),
            T(2.09518325850772),
            T(2.74847072498540),
            T(3.46265693360227),
            T(4.30444857047363)};

  if (n == 15u)
    return {T(0.00000000000000),
            T(0.56506958325558),
            T(1.13611558521092),
            T(1.71999257518649),
            T(2.32573248617386),
            T(2.96716692790560),
            T(3.66995037340445),
            T(4.49999070730939)};

  if (n == 16u)
    return {T(0.27348104613815),
            T(0.82295144914466),
            T(1.38025853919888),
            T(1.95178799091625),
            T(2.54620215784748),
            T(3.17699916197996),
            T(3.86944790486012),
            T(4.68873893930582)};

  if (n == 17u)
    return {T(0),
            T(0.5316330013427),
            T(1.0676487257435),
            T(1.6129243142212),
            T(2.1735028266666),
            T(2.7577629157039),
            T(3.3789320911415),
            T(4.0619466758755),
            T(4.8713451936744)};
  if (n == 18u)
    return {T(0.2582677505191),
            T(0.7766829192674),
            T(1.3009208583896),
            T(1.8355316042616),
            T(2.3862990891667),
            T(2.9613775055316),
            T(3.5737690684863),
            T(4.2481178735681),
            T(5.0483640088745)};
  if (n == 19u)
    return {T(0),
            T(0.5035201634239),
            T(1.0103683871343),
            T(1.5241706193935),
            T(2.0492317098506),
            T(2.5911337897945),
            T(3.1578488183476),
            T(3.7621873519640),
            T(4.4285328066038),
            T(5.2202716905375)};
  if (n == 20u)
    return {T(0.2453407083009),
            T(0.7374737285454),
            T(1.2340762153953),
            T(1.7385377121166),
            T(2.2549740020893),
            T(2.7888060584281),
            T(3.347854567332),
            T(3.9447640401156),
            T(4.6036824495507),
            T(5.3874808900112)};

  return {};
}

/// \param [in] Tolerance of the root. This value must be smaller than
/// the smallest difference between adjacent roots in the given range
/// with n <= 20.
template <class T> void testHermiteRoots(T Tolerance) {
  for (unsigned n = 0; n <= 20u; ++n) {
    const auto Roots = getHermiteRoots<T>(n);
    for (T x : Roots) {
      // the roots are symmetric: if x is a root, so is -x
      if (x > T(0))
        assert(std::signbit(std::experimental::hermite(n, -x + Tolerance)) !=
               std::signbit(std::experimental::hermite(n, -x - Tolerance)));
      assert(std::signbit(std::experimental::hermite(n, x + Tolerance)) !=
             std::signbit(std::experimental::hermite(n, x - Tolerance)));
    }
  }
}

template <class T>
void testHermite(const T AbsTolerance, const T RelTolerance) {
  testHermiteNaNPropagation<T>();
  const T Samples[] = {T(-1.0), T(-0.5), T(-0.1), T(0.0),
                       T(0.1),  T(0.5),  T(1.0)};

  for (T x : Samples) {
    testHermiteNotNaN(x);
    testHermiteAnalytic(x, AbsTolerance, RelTolerance);
  }
}

#endif

int main(int, char **) {
#if _LIBCPP_STD_VER > 14
  testHermite<float>(1e-6f, 1e-6f);
  testHermite<double>(1e-9, 1e-9);
  testHermite<long double>(1e-12l, 1e-12l);

  testRecurrenceRelation<float>(1e-6f, 1e-6f);
  testRecurrenceRelation<double>(1e-9, 1e-9);
  testRecurrenceRelation<long double>(1e-12l, 1e-12l);

  testHermiteRoots<float>(1e-6f);
  testHermiteRoots<double>(1e-9);
  testHermiteRoots<long double>(1e-10l);
#endif
  return 0;
}
