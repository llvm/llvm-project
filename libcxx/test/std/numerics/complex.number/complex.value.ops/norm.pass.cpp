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
//   norm(const complex<T>& x);

#include <complex>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test()
{
    std::complex<T> z(3, 4);
    assert(norm(z) == 25);
}

template <class T>
void test_edges() {
  const unsigned N = sizeof(testcases<T>) / sizeof(testcases<T>[0]);
  for (unsigned i = 0; i < N; ++i) {
    T r = norm(testcases<T>[i]);
    switch (classify(testcases<T>[i])) {
    case zero:
      assert(r == 0);
      assert(!std::signbit(r));
      break;
    case non_zero:
      assert(std::isfinite(r) && r > 0);
      break;
    case lowest_value:
    case maximum_value:
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
