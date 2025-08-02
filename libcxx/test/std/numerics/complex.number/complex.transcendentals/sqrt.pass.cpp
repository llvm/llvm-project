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
//   sqrt(const complex<T>& x);

#include <complex>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test(const std::complex<T>& c, std::complex<T> x)
{
    std::complex<T> a = sqrt(c);
    is_about(real(a), real(x));
    assert(std::abs(imag(c)) < 1.e-6);
}

template <class T>
void
test()
{
    test(std::complex<T>(64, 0), std::complex<T>(8, 0));
}

template<class T>
bool in_range(const T value, const T low, const T high) { return low < value && value < high; }

// cosine calculations for std::arg terminate at different times based on the precision of T,
// and can produce a very small negative number instead of a very small positive one, so we
// should actually check that we're really close to zero instead of positive.
bool approaching_zero(const float value) {
    return value == -0.0f || in_range(value, -4.38e-5f, 4.38e-5f);
}

bool approaching_zero(double) { return false; }

bool approaching_zero(const long double value) {
    return value == -0.0f || in_range(value, -3.0e-17L, 3.0e-17L);
}

template <class T>
void test_edges() {
  const unsigned N = sizeof(testcases<T>) / sizeof(testcases<T>[0]);
  for (unsigned i = 0; i < N; ++i) {
    std::complex<T> r = sqrt(testcases<T>[i]);
    if (testcases<T>[i].real() == 0 && testcases<T>[i].imag() == 0) {
      assert(!std::signbit(r.real()) || approaching_zero(r.real()));
      assert(std::signbit(r.imag()) == std::signbit(testcases<T>[i].imag()));
    } else if (std::isinf(testcases<T>[i].imag())) {
      assert(std::isinf(r.real()));
      assert(r.real() > 0);
      assert(std::isinf(r.imag()));
      assert(std::signbit(r.imag()) == std::signbit(testcases<T>[i].imag()));
    } else if (std::isfinite(testcases<T>[i].real()) && std::isnan(testcases<T>[i].imag())) {
      assert(std::isnan(r.real()));
      assert(std::isnan(r.imag()));
    } else if (std::isinf(testcases<T>[i].real()) && testcases<T>[i].real() < 0 &&
               std::isfinite(testcases<T>[i].imag())) {
      assert(r.real() == 0);
      assert(!std::signbit(r.real()));
      assert(std::isinf(r.imag()));
      assert(std::signbit(testcases<T>[i].imag()) == std::signbit(r.imag()));
    } else if (std::isinf(testcases<T>[i].real()) && testcases<T>[i].real() > 0 &&
               std::isfinite(testcases<T>[i].imag())) {
      assert(std::isinf(r.real()));
      assert(r.real() > 0);
      assert(r.imag() == 0);
      assert(std::signbit(testcases<T>[i].imag()) == std::signbit(r.imag()));
    } else if (std::isinf(testcases<T>[i].real()) && testcases<T>[i].real() < 0 && std::isnan(testcases<T>[i].imag())) {
      assert(std::isnan(r.real()));
      assert(std::isinf(r.imag()));
    } else if (std::isinf(testcases<T>[i].real()) && testcases<T>[i].real() > 0 && std::isnan(testcases<T>[i].imag())) {
      assert(std::isinf(r.real()));
      assert(r.real() > 0);
      assert(std::isnan(r.imag()));
    } else if (std::isnan(testcases<T>[i].real()) &&
               (std::isfinite(testcases<T>[i].imag()) || std::isnan(testcases<T>[i].imag()))) {
      assert(std::isnan(r.real()));
      assert(std::isnan(r.imag()));
    } else if (std::signbit(testcases<T>[i].imag())) {
      assert(!std::signbit(r.real()) || approaching_zero(r.real()));
      assert(std::signbit(r.imag()));
    } else {
      assert(!std::signbit(r.real()) || approaching_zero(r.real()));
      assert(!std::signbit(r.imag()));
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
