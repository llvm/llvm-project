//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// Check that functions are marked [[nodiscard]]

#include <complex>
#include <utility>

#include "test_macros.h"

template <typename T>
void test() {
  const std::complex<T> c;
  const std::complex<T> d;

  c.real(); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  c.imag(); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  c + d;    // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  c + T(1); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  T(1) + c; // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  c - d;    // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  c - T(1); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  T(1) - c; // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

#if 0 // TODO: Enable when https://llvm.org/PR171031 is resolved.
  c* d;     // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
  c* T(1);  // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  T(1) * c; // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

#if 0 // TODO: Enable when https://llvm.org/PR171031 is resolved.
  c / d;    // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
  c / T(1); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  T(1) / c; // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  +c; // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  -c; // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::real(c);    // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::imag(c);    // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::real(T(1)); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::imag(T(1)); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::abs(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::arg(c);    // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::arg(T(1)); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::norm(c);    // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::norm(T(1)); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::conj(c);    // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::conj(T(1)); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::proj(c);    // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::proj(T(1)); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  // expected-warning@+1 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::polar(T(93), T(0));

  std::log(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::log10(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::sqrt(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::exp(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::complex<double> ic;

  std::pow(c, c);  // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::pow(c, ic); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::pow(ic, c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::asinh(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::acosh(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::atanh(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::sinh(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::cosh(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::tanh(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::asin(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::acos(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::atan(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::sin(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::cos(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::tan(c); // expected-warning 3 {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 26
  const std::complex<T> cc;

  // expected-warning@+1 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(c);
  // expected-warning@+1 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(std::move(c));
  // expected-warning@+1 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(c);
  // expected-warning@+1 3 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(std::move(cc));
#endif
}

void test() {
  test<float>();
  test<double>();
  test<long double>();
}
