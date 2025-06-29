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
//   sinh(const complex<T>& x);

#include <complex>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test(const std::complex<T>& c, std::complex<T> x)
{
    assert(sinh(c) == x);
}

template <class T>
void
test()
{
    test(std::complex<T>(0, 0), std::complex<T>(0, 0));
}

#include <print>

template <class T>
void test_edges() {
  const unsigned N = sizeof(testcases<T>) / sizeof(testcases<T>[0]);
  for (unsigned i = 0; i < N; ++i) {
    std::complex<T> r = sinh(testcases<T>[i]);
    if (testcases<T>[i].real() == 0 && testcases<T>[i].imag() == 0) {
      assert(r.real() == 0);
      assert(std::signbit(r.real()) == std::signbit(testcases<T>[i].real()));
      assert(r.imag() == 0);
      assert(std::signbit(r.imag()) == std::signbit(testcases<T>[i].imag()));
    } else if (testcases<T>[i].real() == 0 && std::isinf(testcases<T>[i].imag())) {
      assert(r.real() == 0);
      assert(std::isnan(r.imag()));
    } else if (std::isfinite(testcases<T>[i].real()) && std::isinf(testcases<T>[i].imag())) {
      assert(std::isnan(r.real()));
      assert(std::isnan(r.imag()));
    } else if (testcases<T>[i].real() == 0 && std::isnan(testcases<T>[i].imag())) {
      assert(r.real() == 0);
      assert(std::isnan(r.imag()));
    } else if (std::isfinite(testcases<T>[i].real()) && std::isnan(testcases<T>[i].imag())) {
      assert(std::isnan(r.real()));
      assert(std::isnan(r.imag()));
    } else if (std::isinf(testcases<T>[i].real()) && testcases<T>[i].imag() == 0) {
      assert(std::isinf(r.real()));
      assert(std::signbit(r.real()) == std::signbit(testcases<T>[i].real()));
      assert(r.imag() == 0);
      assert(std::signbit(r.imag()) == std::signbit(testcases<T>[i].imag()));
    } else if (std::isinf(testcases<T>[i].real()) && std::isfinite(testcases<T>[i].imag())) {
      assert(std::isinf(r.real()));
      assert(std::signbit(r.real()) == std::signbit(testcases<T>[i].real() * cos(testcases<T>[i].imag())));
      assert(std::isinf(r.imag()));
      assert(std::signbit(r.imag()) == std::signbit(sin(testcases<T>[i].imag())));
    } else if (std::isinf(testcases<T>[i].real()) && std::isinf(testcases<T>[i].imag())) {
      assert(std::isinf(r.real()));
      assert(std::isnan(r.imag()));
    } else if (std::isinf(testcases<T>[i].real()) && std::isnan(testcases<T>[i].imag())) {
      assert(std::isinf(r.real()));
      assert(std::isnan(r.imag()));
    } else if (std::isnan(testcases<T>[i].real()) && testcases<T>[i].imag() == 0) {
      assert(std::isnan(r.real()));
      assert(r.imag() == 0);
      assert(std::signbit(r.imag()) == std::signbit(testcases<T>[i].imag()));
    } else if (std::isnan(testcases<T>[i].real()) && std::isfinite(testcases<T>[i].imag())) {
      assert(std::isnan(r.real()));
      assert(std::isnan(r.imag()));
    } else if (std::isnan(testcases<T>[i].real()) && std::isnan(testcases<T>[i].imag())) {
      assert(std::isnan(r.real()));
      assert(std::isnan(r.imag()));
    } else if (std::isfinite(testcases<T>[i].real()) && std::isfinite(testcases<T>[i].imag())) {
      assert(!std::isnan(r.real()));

      bool const nan_okay = std::isinf(r.real()) && testcases<T>[i].imag() == 0; // inf * 0 == NaN
      assert(!std::isnan(r.imag()) || nan_okay);
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
