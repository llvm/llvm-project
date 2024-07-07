//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

#include <cassert>
#include <cmath>
#include <limits>
#include <vector>
#include <array>

#include <assert_macros.h>

namespace {

inline constexpr unsigned MAX_N = 128;

template <class T>
std::array<T, 7> sample_points() {
  return {-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0};
}

template <class T>
void test_NaN_propagation() {
  T NaN = std::numeric_limits<T>::quiet_NaN();
  for (unsigned n = 0; n < MAX_N; ++n)
    assert(std::isnan(std::hermite(n, NaN)));
}

template <class T>
void test_not_NaN(T x) {
  assert(!std::isnan(x));
  for (unsigned n = 0; n < MAX_N; ++n)
    assert(!std::isnan(std::hermite(n, x)));
}

template <class T>
struct CompareFloatingValues {
  T abs_tol;
  T rel_tol;

  bool operator()(T result, T expected) const {
    if (std::isinf(expected) && std::isinf(result))
      return true;

    if (std::isnan(expected) || std::isnan(result))
      return false;

    T tol = abs_tol + std::abs(expected) * rel_tol;
    return std::abs(result - expected) < tol;
  }
};

template <class T>
void test_analytic_solution(T x, T abs_tol, T rel_tol) {
  assert(!std::isnan(x));

  const auto h0 = [](T) -> T { return 1; };
  const auto h1 = [](T y) -> T { return 2 * y; };
  const auto h2 = [](T y) -> T { return 4 * y * y - 2; };
  const auto h3 = [](T y) -> T { return y * (8 * y * y - 12); };
  const auto h4 = [](T y) -> T { return (16 * std::pow(y, 4) - 48 * y * y + 12); };
  const auto h5 = [](T y) -> T { return y * (32 * std::pow(y, 4) - 160 * y * y + 120); };

  const CompareFloatingValues<T> compare{.abs_tol = abs_tol, .rel_tol = rel_tol};
  assert(compare(std::hermite(0, x), h0(x)));
  assert(compare(std::hermite(1, x), h1(x)));
  assert(compare(std::hermite(2, x), h2(x)));
  assert(compare(std::hermite(3, x), h3(x)));
  assert(compare(std::hermite(4, x), h4(x)));
  assert(compare(std::hermite(5, x), h5(x)));
}

/// \details This method checks if the following recurrence relation holds:
/// \f[
///   H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
/// \f]
template <class T>
void test_recurrence_relation(T x, T abs_tol, T rel_tol) {
  assert(!std::isnan(x));

  const CompareFloatingValues<T> compare{.abs_tol = abs_tol, .rel_tol = rel_tol};
  for (unsigned n = 1; n < MAX_N - 1; ++n) {
    T H_next            = std::hermite(n + 1, x);
    T H_next_recurrence = 2 * (x * std::hermite(n, x) - n * std::hermite(n - 1, x));

    if (std::isinf(H_next))
      break;
    assert(compare(H_next, H_next_recurrence));
  }
}

template <class T>
void test_recurrence_relation(T abs_tol, T rel_tol) {
  for (T x : sample_points<T>())
    test_recurrence_relation(x, abs_tol, rel_tol);
}

/// \note Roots are taken from
/// Salzer, Herbert E., Ruth Zucker, and Ruth Capuano.
/// Table of the zeros and weight factors of the first twenty Hermite
/// polynomials. US Government Printing Office, 1952.
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

  default:
    throw "Roots of Hermite polynomial of order " + std::to_string(n) + " not implemented!\n";
  }
}

/// \param [in] Tolerance of the root. This value must be smaller than
/// the smallest difference between adjacent roots in the given range
/// with n <= 20.
template <class T>
void test_roots(T Tolerance) {
  const auto is_sign_change = [Tolerance](unsigned n, T x) -> bool {
    return std::hermite(n, x - Tolerance) * std::hermite(n, x + Tolerance) < 0;
  };

  for (unsigned n = 0; n <= 20u; ++n) {
    for (T x : get_roots<T>(n)) {
      // the roots are symmetric: if x is a root, so is -x
      if (x > 0)
        assert(is_sign_change(n, -x));
      assert(is_sign_change(n, x));
    }
  }
}

template <class T>
void test_hermite(T abs_tol, T rel_tol) {
  test_NaN_propagation<T>();

  for (T x : sample_points<T>()) {
    test_not_NaN(x);
    test_analytic_solution(x, abs_tol, rel_tol);
  }
}

template <class Integer>
void test_integers() {
  for (unsigned n = 0; n < MAX_N; ++n)
    for (Integer x : {-1, 0, 1})
      assert(std::hermite(n, x) == std::hermite(n, static_cast<double>(x)));
}

void test_hermitef() {
  for (unsigned n = 0; n < MAX_N; ++n)
    for (float x : sample_points<float>())
      assert(std::hermite(n, x) == std::hermitef(n, x));
}

void test_hermitel() {
  for (unsigned n = 0; n < MAX_N; ++n)
    for (long double x : sample_points<long double>())
      assert(std::hermite(n, x) == std::hermitel(n, x));
}
} // namespace

int main(int, char**) {
  test_hermite<float>(1e-5f, 1e-5f);
  test_hermite<double>(1e-11, 1e-11);
  test_hermite<long double>(1e-12l, 1e-12l);

  test_hermitef();
  test_hermitel();

  test_recurrence_relation<float>(1e-5f, 1e-5f);
  test_recurrence_relation<double>(1e-11, 1e-11);
  test_recurrence_relation<long double>(1e-12l, 1e-12l);

  test_roots<float>(1e-5f);
  test_roots<double>(1e-9);
  test_roots<long double>(1e-10l);

  test_integers<short>();
  test_integers<int>();
  test_integers<long>();
  test_integers<long long>();

  return 0;
}
