//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool isunordered(floating-point-type x, floating-point-type y); // constexpr since C++23

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

#include "test_macros.h"
#include "type_algorithms.h"

struct TestFloat {
  template <class T>
  static TEST_CONSTEXPR_CXX23 bool test() {
    assert(!std::isunordered(T(1), T(2)));
    assert(!std::isunordered(T(1), T(1)));

    assert(std::isunordered(std::numeric_limits<T>::quiet_NaN(), T(0)));
    assert(std::isunordered(T(0), std::numeric_limits<T>::quiet_NaN()));
    assert(std::isunordered(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()));

    assert(std::isunordered(std::numeric_limits<T>::signaling_NaN(), T(0)));
    assert(!std::isunordered(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity()));
    assert(!std::isunordered(std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()));

    return true;
  }

  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() {
    test<T>();
#if TEST_STD_VER >= 23
    static_assert(test<T>());
#endif
  }
};

struct TestInt {
  template <class T>
  static TEST_CONSTEXPR_CXX23 bool test() {
    assert(!std::isunordered(T(1), T(2)));
    assert(!std::isunordered(T(1), T(1)));
    assert(!std::isunordered(std::numeric_limits<T>::max(), T(0)));
    assert(!std::isunordered(std::numeric_limits<T>::max(), std::numeric_limits<T>::max()));

    return true;
  }

  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() {
    test<T>();
#if TEST_STD_VER >= 23
    static_assert(test<T>());
#endif
  }
};

int main(int, char**) {
  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::integral_types(), TestInt());

  return 0;
}
