//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   complex<T>
//   acosh(const complex<T>& x);

#include <complex>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test(const std::complex<T>& c, std::complex<T> x)
{
    assert(acosh(c) == x);
}

template <class T>
void
test()
{
    test(std::complex<T>(INFINITY, 1), std::complex<T>(INFINITY, 0));
}

template <class T>
void test_edges() {
  const T pi       = std::atan2(+0., -0.);
  const unsigned N = sizeof(testcases<T>) / sizeof(testcases<T>[0]);
  for (unsigned i = 0; i < N; ++i) {
    std::complex<T> r = acosh(testcases<T>[i]);
    if (testcases<T>[i].real() == 0 && testcases<T>[i].imag() == 0) {
      assert(!std::signbit(r.real()));
      if (std::signbit(testcases<T>[i].imag()))
        is_about(r.imag(), -pi / 2);
      else
        is_about(r.imag(), pi / 2);
    } else if (testcases<T>[i].real() == 1 && testcases<T>[i].imag() == 0) {
      assert(r.real() == 0);
      assert(!std::signbit(r.real()));
      assert(r.imag() == 0);
      assert(std::signbit(r.imag()) == std::signbit(testcases<T>[i].imag()));
    } else if (testcases<T>[i].real() == -1 && testcases<T>[i].imag() == 0) {
      assert(r.real() == 0);
      assert(!std::signbit(r.real()));
      if (std::signbit(testcases<T>[i].imag()))
        is_about(r.imag(), -pi);
      else
        is_about(r.imag(), pi);
    } else if (std::isfinite(testcases<T>[i].real()) && std::isinf(testcases<T>[i].imag())) {
      assert(std::isinf(r.real()));
      assert(r.real() > 0);
      if (std::signbit(testcases<T>[i].imag()))
        is_about(r.imag(), -pi / 2);
      else
        is_about(r.imag(), pi / 2);
    } else if (std::isfinite(testcases<T>[i].real()) && std::isnan(testcases<T>[i].imag())) {
      assert(std::isnan(r.real()));
      assert(std::isnan(r.imag()));
    } else if (std::isinf(testcases<T>[i].real()) && testcases<T>[i].real() < 0 &&
               std::isfinite(testcases<T>[i].imag())) {
      assert(std::isinf(r.real()));
      assert(r.real() > 0);
      if (std::signbit(testcases<T>[i].imag()))
        is_about(r.imag(), -pi);
      else
        is_about(r.imag(), pi);
    } else if (std::isinf(testcases<T>[i].real()) && testcases<T>[i].real() > 0 &&
               std::isfinite(testcases<T>[i].imag())) {
      assert(std::isinf(r.real()));
      assert(r.real() > 0);
      assert(r.imag() == 0);
      assert(std::signbit(r.imag()) == std::signbit(testcases<T>[i].imag()));
    } else if (std::isinf(testcases<T>[i].real()) && testcases<T>[i].real() < 0 && std::isinf(testcases<T>[i].imag())) {
      assert(std::isinf(r.real()));
      assert(r.real() > 0);
      if (std::signbit(testcases<T>[i].imag()))
        is_about(r.imag(), T(-0.75) * pi);
      else
        is_about(r.imag(), T(0.75) * pi);
    } else if (std::isinf(testcases<T>[i].real()) && testcases<T>[i].real() > 0 && std::isinf(testcases<T>[i].imag())) {
      assert(std::isinf(r.real()));
      assert(r.real() > 0);
      if (std::signbit(testcases<T>[i].imag()))
        is_about(r.imag(), T(-0.25 * pi));
      else
        is_about(r.imag(), T(0.25 * pi));
    } else if (std::isinf(testcases<T>[i].real()) && std::isnan(testcases<T>[i].imag())) {
      assert(std::isinf(r.real()));
      assert(r.real() > 0);
      assert(std::isnan(r.imag()));
    } else if (std::isnan(testcases<T>[i].real()) && std::isfinite(testcases<T>[i].imag())) {
      assert(std::isnan(r.real()));
      assert(std::isnan(r.imag()));
    } else if (std::isnan(testcases<T>[i].real()) && std::isinf(testcases<T>[i].imag())) {
      assert(std::isinf(r.real()));
      assert(r.real() > 0);
      assert(std::isnan(r.imag()));
    } else if (std::isnan(testcases<T>[i].real()) && std::isnan(testcases<T>[i].imag())) {
      assert(std::isnan(r.real()));
      assert(std::isnan(r.imag()));
    } else {
      assert(!std::signbit(r.real()));
      assert(std::signbit(r.imag()) == std::signbit(testcases<T>[i].imag()));
    }
  }
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();
    test_edges<float>();
    test_edges<double>();
    test_edges<long double>();

    return 0;
}
