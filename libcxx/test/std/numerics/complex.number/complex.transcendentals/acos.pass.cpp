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
//   acos(const complex<T>& x);

#include <complex>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test(const std::complex<T>& c, std::complex<T> x)
{
    assert(acos(c) == x);
}

template <class T>
void
test()
{
    test(std::complex<T>(INFINITY, 1), std::complex<T>(0, -INFINITY));
}

template <class T>
void test_edges() {
  const T pi       = std::atan2(+0., -0.);
  const unsigned N = sizeof(testcases<T>) / sizeof(testcases<T>[0]);
  for (unsigned i = 0; i < N; ++i) {
    std::complex<T> r = acos(testcases<T>[i]);
    if (testcases<T>[i].real() == 0 && testcases<T>[i].imag() == 0) {
      is_about(r.real(), pi / 2);
      assert(r.imag() == 0);
      assert(std::signbit(testcases<T>[i].imag()) != std::signbit(r.imag()));
    } else if (testcases<T>[i].real() == 0 && std::isnan(testcases<T>[i].imag())) {
      is_about(r.real(), pi / 2);
      assert(std::isnan(r.imag()));
    } else if (std::isfinite(testcases<T>[i].real()) && std::isinf(testcases<T>[i].imag())) {
      is_about(r.real(), pi / 2);
      assert(std::isinf(r.imag()));
      assert(std::signbit(testcases<T>[i].imag()) != std::signbit(r.imag()));
    } else if (std::isfinite(testcases<T>[i].real()) && testcases<T>[i].real() != 0 &&
               std::isnan(testcases<T>[i].imag())) {
      assert(std::isnan(r.real()));
      assert(std::isnan(r.imag()));
    } else if (std::isinf(testcases<T>[i].real()) && testcases<T>[i].real() < 0 &&
               std::isfinite(testcases<T>[i].imag())) {
      is_about(r.real(), pi);
      assert(std::isinf(r.imag()));
      assert(std::signbit(testcases<T>[i].imag()) != std::signbit(r.imag()));
    } else if (std::isinf(testcases<T>[i].real()) && testcases<T>[i].real() > 0 &&
               std::isfinite(testcases<T>[i].imag())) {
      assert(r.real() == 0);
      assert(!std::signbit(r.real()));
      assert(std::isinf(r.imag()));
      assert(std::signbit(testcases<T>[i].imag()) != std::signbit(r.imag()));
    } else if (std::isinf(testcases<T>[i].real()) && testcases<T>[i].real() < 0 && std::isinf(testcases<T>[i].imag())) {
      is_about(r.real(), T(0.75) * pi);
      assert(std::isinf(r.imag()));
      assert(std::signbit(testcases<T>[i].imag()) != std::signbit(r.imag()));
    } else if (std::isinf(testcases<T>[i].real()) && testcases<T>[i].real() > 0 && std::isinf(testcases<T>[i].imag())) {
      is_about(r.real(), T(0.25) * pi);
      assert(std::isinf(r.imag()));
      assert(std::signbit(testcases<T>[i].imag()) != std::signbit(r.imag()));
    } else if (std::isinf(testcases<T>[i].real()) && std::isnan(testcases<T>[i].imag())) {
      assert(std::isnan(r.real()));
      assert(std::isinf(r.imag()));
    } else if (std::isnan(testcases<T>[i].real()) && std::isfinite(testcases<T>[i].imag())) {
      assert(std::isnan(r.real()));
      assert(std::isnan(r.imag()));
    } else if (std::isnan(testcases<T>[i].real()) && std::isinf(testcases<T>[i].imag())) {
      assert(std::isnan(r.real()));
      assert(std::isinf(r.imag()));
      assert(std::signbit(testcases<T>[i].imag()) != std::signbit(r.imag()));
    } else if (std::isnan(testcases<T>[i].real()) && std::isnan(testcases<T>[i].imag())) {
      assert(std::isnan(r.real()));
      assert(std::isnan(r.imag()));
    } else if (!std::signbit(testcases<T>[i].real()) && !std::signbit(testcases<T>[i].imag())) {
      assert(!std::signbit(r.real()));
      assert(std::signbit(r.imag()));
    } else if (std::signbit(testcases<T>[i].real()) && !std::signbit(testcases<T>[i].imag())) {
      assert(!std::signbit(r.real()));
      assert(std::signbit(r.imag()));
    } else if (std::signbit(testcases<T>[i].real()) && std::signbit(testcases<T>[i].imag())) {
      assert(!std::signbit(r.real()));
      assert(!std::signbit(r.imag()));
    } else if (!std::signbit(testcases<T>[i].real()) && std::signbit(testcases<T>[i].imag())) {
      assert(!std::signbit(r.real()));
      assert(!std::signbit(r.imag()));
    } else {
      assert(!std::isnan(r.real()));
      assert(!std::isnan(r.imag()));
      assert(std::signbit(r.real()) == std::signbit(testcases<T>[i].real()));
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
