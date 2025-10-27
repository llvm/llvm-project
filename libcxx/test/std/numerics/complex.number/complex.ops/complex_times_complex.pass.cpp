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
//   operator*(const complex<T>& lhs, const complex<T>& rhs); // constexpr in C++20

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=2131685

#include <complex>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
TEST_CONSTEXPR_CXX20
bool
test()
{
    const std::complex<T> lhs(1.5, 2.5);
    const std::complex<T> rhs(1.5, 2.5);
    assert(lhs * rhs == std::complex<T>(-4.0, 7.5));
    return true;
}

// test edges

template <class T>
TEST_CONSTEXPR_CXX20 bool test_edges() {
  const unsigned N = sizeof(testcases<T>) / sizeof(testcases<T>[0]);
  int classification[N];
  for (unsigned i = 0; i < N; ++i)
    classification[i] = classify(testcases<T>[i]);

  for (unsigned i = 0; i < N; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      std::complex<T> r = testcases<T>[i] * testcases<T>[j];
      switch (classification[i]) {
      case zero:
        switch (classification[j]) {
        case lowest_value:
        case maximum_value:
          continue; // not tested
        case zero:
        case non_zero:
          assert(classify(r) == zero);
          break;
        case inf:
        case NaN:
        case non_zero_nan:
          assert(classify(r) == NaN);
          break;
        }
        break;
      case lowest_value:
      case maximum_value:
        continue; // not tested
      case non_zero:
        switch (classification[j]) {
        case zero:
          assert(classify(r) == zero);
          break;
        case lowest_value:
        case maximum_value:
          continue; // not tested
        case non_zero:
          assert(classify(r) == non_zero);
          break;
        case inf:
          assert(classify(r) == inf);
          break;
        case NaN:
        case non_zero_nan:
          assert(classify(r) == NaN);
          break;
        }
        break;
      case inf:
        switch (classification[j]) {
        case zero:
        case NaN:
          assert(classify(r) == NaN);
          break;
        case lowest_value:
        case maximum_value:
          continue; // not tested
        case non_zero:
        case inf:
        case non_zero_nan:
          assert(classify(r) == inf);
          break;
        }
        break;
      case NaN:
        assert(classify(r) == NaN);
        break;
      case non_zero_nan:
        switch (classification[j]) {
        case inf:
          assert(classify(r) == inf);
          break;
        case lowest_value:
        case maximum_value:
          continue; // not tested
        case zero:
        case non_zero:
        case NaN:
        case non_zero_nan:
          assert(classify(r) == NaN);
          break;
        }
        break;
      }
    }
  }
  return true;
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();
    test_edges<float>();
    test_edges<double>();
    test_edges<long double>();

#if TEST_STD_VER > 17
    static_assert(test<float>());
    static_assert(test<double>());
    static_assert(test<long double>());
    static_assert(test_edges<float>());
    static_assert(test_edges<double>());
    static_assert(test_edges<long double>());
#endif

  return 0;
}
