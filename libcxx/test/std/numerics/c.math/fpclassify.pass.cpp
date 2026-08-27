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
  TEST_CONSTEXPR_CXX23 void operator()() const {
    using lim = std::numeric_limits<T>;

    assert(std::fpclassify(lim::quiet_NaN()) == FP_NAN);
    assert(std::fpclassify(lim::signaling_NaN()) == FP_NAN);

    assert(std::fpclassify(lim::infinity()) == FP_INFINITE);
    assert(std::fpclassify(-lim::infinity()) == FP_INFINITE);

    assert(std::fpclassify(T(1)) == FP_NORMAL);
    assert(std::fpclassify(T(-1)) == FP_NORMAL);
    assert(std::fpclassify(lim::max()) == FP_NORMAL);
    assert(std::fpclassify(lim::min()) == FP_NORMAL);

    assert(std::fpclassify(lim::denorm_min()) == FP_SUBNORMAL);

    assert(std::fpclassify(T(0)) == FP_ZERO);
    assert(std::fpclassify(T(-0.0)) == FP_ZERO);
  }
};

struct TestInt {
  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() const {
    using lim = std::numeric_limits<T>;

    assert(std::fpclassify(T(0)) == FP_ZERO);
    assert(std::fpclassify(T(1)) == FP_NORMAL);
    assert(std::fpclassify(lim::max()) == FP_NORMAL);

    if (std::is_signed<T>::value) {
      assert(std::fpclassify(lim::lowest()) == FP_NORMAL);
      assert(std::fpclassify(T(-1)) == FP_NORMAL);
    }
  }
};

TEST_CONSTEXPR_CXX23 bool test() {
  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::integral_types(), TestInt());

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif
  return 0;
}
