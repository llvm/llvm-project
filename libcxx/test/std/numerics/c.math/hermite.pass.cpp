//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <cmath>

// double         hermite(unsigned n, double x);
// float          hermite(unsigned n, float x);
// long double    hermite(unsigned n, long double x);
// float          hermitef(unsigned n, float x);
// long double    hermitel(unsigned n, long double x);
// template <class Integer>
// double         hermite(unsigned n, Integer x);

#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

#include "type_algorithms.h"

template <class Real>
constexpr unsigned get_maximal_order() {
  if constexpr (std::numeric_limits<Real>::is_iec559)
    return 128;
  else { // Workaround for z/OS HexFloat.
    // Note |H_n(x)| < 10^75 for n < 39 and x in sample_points().
    static_assert(std::numeric_limits<Real>::max_exponent10 == 75);
    return 39;
  }
}

template <class T>
std::array<T, 11> sample_points() {
  return {-12.34, -7.42, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 5.67, 15.67};
}

template <class Real>
class CompareFloatingValues {
private:
  Real abs_tol;
  Real rel_tol;

public:
  CompareFloatingValues() {
    abs_tol = []() -> Real {
      if (std::is_same_v<Real, float>)
        return 1e-5f;
      else if (std::is_same_v<Real, double>)
        return 1e-11;
      else
        return 1e-12l;
    }();

    rel_tol = abs_tol;
  }

  bool operator()(Real result, Real expected) const {
    if (std::isinf(expected) && std::isinf(result))
      return result == expected;

    if (std::isnan(expected) || std::isnan(result))
      return false;

    Real tol = abs_tol + std::abs(expected) * rel_tol;
    return std::abs(result - expected) < tol;
  }
};

// Roots are taken from
// Salzer, Herbert E., Ruth Zucker, and Ruth Capuano.
// Table of the zeros and weight factors of the first twenty Hermite
// polynomials. US Government Printing Office, 1952.
template <class T>
std::vector<T> get_roots(unsigned n) {
  switch (n) {
  case 0:
    return {};
  case 1:
    return {T(0)};
  case 2:
    return {T(0.707106781186548)};
  case 3:
    return {T(0), T(1.224744871391589)};
  case 4:
    return {T(0.524647623275290), T(1.650680123885785)};
  case 5:
    return {T(0), T(0.958572464613819), T(2.020182870456086)};
  case 6:
    return {T(0.436077411927617), T(1.335849074013697), T(2.350604973674492)};
  case 7:
    return {T(0), T(0.816287882858965), T(1.673551628767471), T(2.651961356835233)};
  case 8:
    return {T(0.381186990207322), T(1.157193712446780), T(1.981656756695843), T(2.930637420257244)};
  case 9:
    return {T(0), T(0.723551018752838), T(1.468553289216668), T(2.266580584531843), T(3.190993201781528)};
  case 10:
    return {
        T(0.342901327223705), T(1.036610829789514), T(1.756683649299882), T(2.532731674232790), T(3.436159118837738)};
  case 11:
    return {T(0),
            T(0.65680956682100),
            T(1.326557084494933),
            T(2.025948015825755),
            T(2.783290099781652),
            T(3.668470846559583)};

  case 12:
    return {T(0.314240376254359),
            T(0.947788391240164),
            T(1.597682635152605),
            T(2.279507080501060),
            T(3.020637025120890),
            T(3.889724897869782)};

  case 13:
    return {T(0),
            T(0.605763879171060),
            T(1.220055036590748),
            T(1.853107651601512),
            T(2.519735685678238),
            T(3.246608978372410),
            T(4.101337596178640)};

  case 14:
    return {T(0.29174551067256),
            T(0.87871378732940),
            T(1.47668273114114),
            T(2.09518325850772),
            T(2.74847072498540),
            T(3.46265693360227),
            T(4.30444857047363)};

  case 15:
    return {T(0.00000000000000),
            T(0.56506958325558),
            T(1.13611558521092),
            T(1.71999257518649),
            T(2.32573248617386),
            T(2.96716692790560),
            T(3.66995037340445),
            T(4.49999070730939)};

  case 16:
    return {T(0.27348104613815),
            T(0.82295144914466),
            T(1.38025853919888),
            T(1.95178799091625),
            T(2.54620215784748),
            T(3.17699916197996),
            T(3.86944790486012),
            T(4.68873893930582)};

  case 17:
    return {T(0),
            T(0.5316330013427),
            T(1.0676487257435),
            T(1.6129243142212),
            T(2.1735028266666),
            T(2.7577629157039),
            T(3.3789320911415),
            T(4.0619466758755),
            T(4.8713451936744)};

  case 18:
    return {T(0.2582677505191),
            T(0.7766829192674),
            T(1.3009208583896),
            T(1.8355316042616),
            T(2.3862990891667),
            T(2.9613775055316),
            T(3.5737690684863),
            T(4.2481178735681),
            T(5.0483640088745)};

  case 19:
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

  case 20:
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

  default: // polynom degree n>20 is unsupported
    assert(false);
    return {T(-42)};
  }
}

template <class Real>
void test() {
  if constexpr (
      std::numeric_limits<Real>::has_quiet_NaN &&
      std::numeric_limits<
          Real>::has_signaling_NaN) { // checks if NaNs are reported correctly (i.e. output == input for input == NaN)
    using nl = std::numeric_limits<Real>;
    for (Real NaN : {nl::quiet_NaN(), nl::signaling_NaN()})
      for (unsigned n = 0; n < get_maximal_order<Real>(); ++n)
        assert(std::isnan(std::hermite(n, NaN)));
  }

  if constexpr (std::numeric_limits<Real>::has_quiet_NaN &&
                std::numeric_limits<
                    Real>::has_signaling_NaN) { // simple sample points for n=0..127 should not produce NaNs.
    for (Real x : sample_points<Real>())
      for (unsigned n = 0; n < get_maximal_order<Real>(); ++n)
        assert(!std::isnan(std::hermite(n, x)));
  }

  { // checks std::hermite(n, x) for n=0..5 against analytic polynoms
    const auto h0 = [](Real) -> Real { return 1; };
    const auto h1 = [](Real y) -> Real { return 2 * y; };
    const auto h2 = [](Real y) -> Real { return 4 * y * y - 2; };
    const auto h3 = [](Real y) -> Real { return y * (8 * y * y - 12); };
    const auto h4 = [](Real y) -> Real { return (16 * std::pow(y, 4) - 48 * y * y + 12); };
    const auto h5 = [](Real y) -> Real { return y * (32 * std::pow(y, 4) - 160 * y * y + 120); };

    for (Real x : sample_points<Real>()) {
      const CompareFloatingValues<Real> compare;
      assert(compare(std::hermite(0, x), h0(x)));
      assert(compare(std::hermite(1, x), h1(x)));
      assert(compare(std::hermite(2, x), h2(x)));
      assert(compare(std::hermite(3, x), h3(x)));
      assert(compare(std::hermite(4, x), h4(x)));
      assert(compare(std::hermite(5, x), h5(x)));
    }
  }

  { // checks std::hermitef for bitwise equality with std::hermite(unsigned, float)
    if constexpr (std::is_same_v<Real, float>)
      for (unsigned n = 0; n < get_maximal_order<Real>(); ++n)
        for (float x : sample_points<float>())
          assert(std::hermite(n, x) == std::hermitef(n, x));
  }

  { // checks std::hermitel for bitwise equality with std::hermite(unsigned, long double)
    if constexpr (std::is_same_v<Real, long double>)
      for (unsigned n = 0; n < get_maximal_order<Real>(); ++n)
        for (long double x : sample_points<long double>())
          assert(std::hermite(n, x) == std::hermitel(n, x));
  }

  { // Checks if the characteristic recurrence relation holds:    H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    for (Real x : sample_points<Real>()) {
      for (unsigned n = 1; n < get_maximal_order<Real>() - 1; ++n) {
        Real H_next            = std::hermite(n + 1, x);
        Real H_next_recurrence = 2 * (x * std::hermite(n, x) - n * std::hermite(n - 1, x));

        if (std::isinf(H_next))
          break;
        const CompareFloatingValues<Real> compare;
        assert(compare(H_next, H_next_recurrence));
      }
    }
  }

  { // sanity checks: hermite polynoms need to change signs at (simple) roots. checked upto order n<=20.

    // root tolerance: must be smaller than the smallest difference between adjacent roots
    Real tol = []() -> Real {
      if (std::is_same_v<Real, float>)
        return 1e-5f;
      else if (std::is_same_v<Real, double>)
        return 1e-9;
      else
        return 1e-10l;
    }();

    const auto is_sign_change = [tol](unsigned n, Real x) -> bool {
      return std::hermite(n, x - tol) * std::hermite(n, x + tol) < 0;
    };

    for (unsigned n = 0; n <= 20u; ++n) {
      for (Real x : get_roots<Real>(n)) {
        // the roots are symmetric: if x is a root, so is -x
        if (x > 0)
          assert(is_sign_change(n, -x));
        assert(is_sign_change(n, x));
      }
    }
  }

  if constexpr (std::numeric_limits<Real>::has_infinity) { // check input infinity is handled correctly
    Real inf = std::numeric_limits<Real>::infinity();
    for (unsigned n = 1; n < get_maximal_order<Real>(); ++n) {
      assert(std::hermite(n, +inf) == inf);
      assert(std::hermite(n, -inf) == ((n & 1) ? -inf : inf));
    }
  }

  if constexpr (std::numeric_limits<
                    Real>::has_infinity) { // check: if overflow occurs that it is mapped to the correct infinity
    if constexpr (std::is_same_v<Real, double>) {
      // Q: Why only double?
      // A: The numeric values (e.g. overflow threshold `n`) below are different for other types.
      static_assert(sizeof(double) == 8);
      for (unsigned n = 0; n < get_maximal_order<Real>(); ++n) {
        // Q: Why n=111 and x=300?
        // A: Both are chosen s.t. the first overflow occurs for some `n<get_maximal_order<Real>()`.
        if (n < 111) {
          assert(std::isfinite(std::hermite(n, +300.0)));
          assert(std::isfinite(std::hermite(n, -300.0)));
        } else {
          double inf = std::numeric_limits<double>::infinity();
          assert(std::hermite(n, +300.0) == inf);
          assert(std::hermite(n, -300.0) == ((n & 1) ? -inf : inf));
        }
      }
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
    // checks that std::hermite(unsigned, Integer) actually wraps std::hermite(unsigned, double)
    for (unsigned n = 0; n < get_maximal_order<double>(); ++n)
      for (Integer x : {-42, -7, -5, -1, 0, 1, 5, 7, 42})
        assert(std::hermite(n, x) == std::hermite(n, static_cast<double>(x)));
  }
};

int main(int, char**) {
  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::type_list<short, int, long, long long>(), TestInt());

  return 0;
}
