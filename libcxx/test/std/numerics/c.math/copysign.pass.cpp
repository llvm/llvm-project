//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool copysign(floating-point-type x, floating-point-type y); // constexpr since C++23

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

#include "test_macros.h"
#include "type_algorithms.h"

struct TestFloat {
  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() const {
    using lim                    = std::numeric_limits<T>;
    TEST_CONSTEXPR_CXX23 T max   = lim::max();
    TEST_CONSTEXPR_CXX23 T min   = lim::min();
    TEST_CONSTEXPR_CXX23 T d_min = lim::denorm_min();
    TEST_CONSTEXPR_CXX23 T inf   = lim::infinity();
    TEST_CONSTEXPR_CXX23 T nan   = lim::quiet_NaN();
    TEST_CONSTEXPR_CXX23 T s_nan = lim::signaling_NaN();

    //assert(std::copysign(nan) == FP_NAN);
    //assert(std::copysign(s_nan) == FP_NAN);

    assert(std::copysign(inf, -inf) == -FP_INFINITE);
    assert(std::copysign(-inf, inf) == FP_INFINITE);

    assert(std::copysign(T(1), T(-1)) == FP_NORMAL);
    assert(std::copysign(T(-1), T(1)) == FP_NORMAL);
    assert(std::copysign(max, -max) == -max);
    assert(std::copysign(-max, max) == max);

    assert(std::copysign(d_min) == FP_SUBNORMAL);

    assert(std::copysign(T(0)) == FP_ZERO);
    assert(std::copysign(T(-0.0)) == FP_ZERO);
  }
};

struct TestInt {
  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() const {
    using lim                  = std::numeric_limits<T>;
    TEST_CONSTEXPR_CXX23 T max = lim::max();
    TEST_CONSTEXPR_CXX23 T low = lim::lowest();

    assert(std::copysign(T(0)) == FP_ZERO);
    assert(std::copysign(T(1)) == FP_NORMAL);
    assert(std::copysign(max) == FP_NORMAL);

    if (std::is_signed<T>::value) {
      assert(std::copysign(low) == FP_NORMAL);
      assert(std::copysign(T(-1)) == FP_NORMAL);
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
