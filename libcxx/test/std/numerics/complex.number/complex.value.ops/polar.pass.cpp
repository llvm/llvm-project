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
//   polar(const T& rho, const T& theta = T());  // changed from '0' by LWG#2870

#include <complex>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test(const T& rho, std::complex<T> x)
{
    assert(std::polar(rho) == x);
}

template <class T>
void
test(const T& rho, const T& theta, std::complex<T> x)
{
    assert(std::polar(rho, theta) == x);
}

template <class T>
void
test()
{
    test(T(0), std::complex<T>(0, 0));
    test(T(1), std::complex<T>(1, 0));
    test(T(100), std::complex<T>(100, 0));
    test(T(0), T(0), std::complex<T>(0, 0));
    test(T(1), T(0), std::complex<T>(1, 0));
    test(T(100), T(0), std::complex<T>(100, 0));
}

template <class T>
void test_edges() {
  const unsigned N = sizeof(testcases<T>) / sizeof(testcases<T>[0]);
  for (unsigned i = 0; i < N; ++i) {
    T r               = real(testcases<T>[i]);
    T theta           = imag(testcases<T>[i]);
    std::complex<T> z = std::polar(r, theta);
    switch (classify(r)) {
    case zero:
      if (std::signbit(r) || classify(theta) == inf || classify(theta) == NaN) {
        int c = classify(z);
        assert(c == NaN || c == non_zero_nan);
      } else {
        assert(z == std::complex<T>());
      }
      break;
    case non_zero:
      if (std::signbit(r) || classify(theta) == inf || classify(theta) == NaN) {
        int c = classify(z);
        assert(c == NaN || c == non_zero_nan);
      } else {
        is_about(std::abs(z), r);
      }
      break;
    case inf:
      if (r < 0) {
        int c = classify(z);
        assert(c == NaN || c == non_zero_nan);
      } else {
        assert(classify(z) == inf);
        if (classify(theta) != NaN && classify(theta) != inf) {
          assert(classify(real(z)) != NaN);
          assert(classify(imag(z)) != NaN);
        }
      }
      break;
    case NaN:
    case non_zero_nan: {
      int c = classify(z);
      assert(c == NaN || c == non_zero_nan);
    } break;
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
