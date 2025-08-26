//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   T
//   arg(const complex<T>& x);

#include <complex>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test()
{
    std::complex<T> z(1, 0);
    assert(arg(z) == 0);
}

template <class T>
void test_edges() {
  const T pi       = std::atan2(T(+0.), T(-0.));
  const unsigned N = sizeof(testcases<T>) / sizeof(testcases<T>[0]);
  for (unsigned i = 0; i < N; ++i) {
    T r = arg(testcases<T>[i]);
    if (std::isnan(testcases<T>[i].real()) || std::isnan(testcases<T>[i].imag()))
      assert(std::isnan(r));
    else {
      switch (classify(testcases<T>[i])) {
      case zero:
        if (std::signbit(testcases<T>[i].real())) {
          if (std::signbit(testcases<T>[i].imag()))
            is_about(r, -pi);
          else
            is_about(r, pi);
        } else {
          assert(std::signbit(testcases<T>[i].imag()) == std::signbit(r));
        }
        break;
      case non_zero:
        if (testcases<T>[i].real() == 0) {
          if (testcases<T>[i].imag() < 0)
            is_about(r, -pi / 2);
          else
            is_about(r, pi / 2);
        } else if (testcases<T>[i].imag() == 0) {
          if (testcases<T>[i].real() < 0) {
            if (std::signbit(testcases<T>[i].imag()))
              is_about(r, -pi);
            else
              is_about(r, pi);
          } else {
            assert(r == 0);
            assert(std::signbit(testcases<T>[i].imag()) == std::signbit(r));
          }
        } else if (testcases<T>[i].imag() > 0)
          assert(r > 0);
        else
          assert(r < 0);
        break;
      case inf:
        if (std::isinf(testcases<T>[i].real()) && std::isinf(testcases<T>[i].imag())) {
          if (testcases<T>[i].real() < 0) {
            if (testcases<T>[i].imag() > 0)
              is_about(r, T(0.75) * pi);
            else
              is_about(r, T(-0.75) * pi);
          } else {
            if (testcases<T>[i].imag() > 0)
              is_about(r, T(0.25) * pi);
            else
              is_about(r, T(-0.25) * pi);
          }
        } else if (std::isinf(testcases<T>[i].real())) {
          if (testcases<T>[i].real() < 0) {
            if (std::signbit(testcases<T>[i].imag()))
              is_about(r, -pi);
            else
              is_about(r, pi);
          } else {
            assert(r == 0);
            assert(std::signbit(r) == std::signbit(testcases<T>[i].imag()));
          }
        } else {
          if (testcases<T>[i].imag() < 0)
            is_about(r, -pi / 2);
          else
            is_about(r, pi / 2);
        }
        break;
      }
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
