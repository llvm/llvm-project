//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool fpclassify(floating-point-type x); // constexpr since C++23

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

#include "test_macros.h"
#include "type_algorithms.h"

struct TestFloat {
  template <class T>
  static TEST_CONSTEXPR_CXX23 bool test() {
    assert(std::fpclassify(std::numeric_limits<T>::quiet_NaN()) == FP_NAN);
    assert(std::fpclassify(std::numeric_limits<T>::signaling_NaN()) == FP_NAN);

    assert(std::fpclassify(std::numeric_limits<T>::infinity()) == FP_INFINITE);
    assert(std::fpclassify(-std::numeric_limits<T>::infinity()) == FP_INFINITE);

    assert(std::fpclassify(T(1)) == FP_NORMAL);
    assert(std::fpclassify(T(-1)) == FP_NORMAL);
    assert(std::fpclassify(std::numeric_limits<T>::max()) == FP_NORMAL);
    assert(std::fpclassify(std::numeric_limits<T>::min()) == FP_NORMAL);

    assert(std::fpclassify(std::numeric_limits<T>::denorm_min()) == FP_SUBNORMAL);

    assert(std::fpclassify(T(0)) == FP_ZERO);
    assert(std::fpclassify(T(-0.0)) == FP_ZERO);

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
    assert(std::fpclassify(T(0)) == FP_ZERO);
    assert(std::fpclassify(T(1)) == FP_NORMAL);
    assert(std::fpclassify(std::numeric_limits<T>::max()) == FP_NORMAL);

    if (std::is_signed<T>::value) {
      assert(std::fpclassify(std::numeric_limits<T>::lowest()) == FP_NORMAL);
      assert(std::fpclassify(T(-1)) == FP_NORMAL);
    }

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
