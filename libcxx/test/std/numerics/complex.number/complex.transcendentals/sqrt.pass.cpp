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

// cosine calculations for std::arg terminate at different times based on the precision of T,
// and can produce a very small negative number instead of a very small positive one, so we
// should actually check that we're really close to zero instead of positive.
template <class T>
bool approaching_zero(const T value) {
  auto in_range = [value](const T low, const T high) { return low < value && value < high; };
  if constexpr (std::is_same_v<T, float>) {
    return value == -0.0f || in_range(-4.38e-5f, 4.38e-5f);
  } else if constexpr (std::is_same_v<T, double>) {
    return false;
  } else if constexpr (std::is_same_v<T, long double>) {
    return value == -0.0f || in_range(-3.0e-17L, 3.0e-17L);
  } else {
    std::abort();
  }
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
      // FIXME(https://github.com/llvm/llvm-project/issues/122172): move `expected_pass` into
      // assert once #122172 is fixed.
      const bool expected_pass = !std::signbit(r.real()) || approaching_zero(r.real());
      assert((i != 156 || std::is_same<T, double>::value) ? expected_pass : !expected_pass);
      assert(std::signbit(r.imag()));
    } else {
      // FIXME(https://github.com/llvm/llvm-project/issues/122172): move `expected_pass` into
      // assert once #122172 is fixed.
      const bool expected_pass = !std::signbit(r.real()) || approaching_zero(r.real());
      assert((i != 155 || std::is_same<T, double>::value) ? expected_pass : !expected_pass);
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
