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
//   abs(const complex<T>& x);

#include <complex>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test()
{
    std::complex<T> z(3, 4);
    assert(abs(z) == 5);
}

template <class T>
void test_edges() {
  const unsigned N = sizeof(testcases<T>) / sizeof(testcases<T>[0]);
  for (unsigned i = 0; i < N; ++i) {
    T r = abs(testcases<T>[i]);
    switch (classify(testcases<T>[i])) {
    case zero:
      assert(r == 0);
      assert(!std::signbit(r));
      break;
    case lowest_value: {
      // It appears that `lowest<float> - relatively_small_number == lowest<float>`, so we check to
      // make sure that abs was actually effective before asserting that it should be infinity.
      bool const ineffective_abs = testcases<T>[i].real() + testcases<T>[i].imag() == -r;
      assert((std::isinf(r) && r > 0) || ineffective_abs);
      break;
    }
    case maximum_value: {
      // It appears that `max<float> + relatively_small_number == max<float>`, so we check to
      // make sure that abs was actually effective before asserting that it should be infinity.
      bool const ineffective_abs = testcases<T>[i].real() + testcases<T>[i].imag() == r;
      assert((std::isinf(r) && r > 0) || ineffective_abs);
      break;
    }
    case non_zero:
      assert(std::isfinite(r) && r > 0);
      break;
    case inf:
      assert(std::isinf(r) && r > 0);
      break;
    case NaN:
      assert(std::isnan(r));
      break;
    case non_zero_nan:
      assert(std::isnan(r));
      break;
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
