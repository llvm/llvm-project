//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// floating-point-type fmax(floating-point-type x, floating-point-type y); // constexpr since C++23

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
    TEST_CONSTEXPR_CXX23 T inf = lim::infinity();
    TEST_CONSTEXPR_CXX23 T nan = lim::quiet_NaN();
    TEST_CONSTEXPR_CXX23 T s_nan = lim::signaling_NaN();

    assert(std::fmax(T(1), T(2)) == T(2));
    assert(std::fmax(T(2), T(1)) == T(2));
    assert(std::fmax(T(-1), T(0)) == T(0));

    assert(std::fmax(inf, T(1)) == inf);
    assert(std::fmax(T(1), inf) == inf);
    assert(std::fmax(-inf, T(1)) == T(1));
    assert(std::fmax(T(1), -inf) == T(1));

    assert(std::fmax(nan, T(1)) == T(1));
    assert(std::fmax(T(1), nan) == T(1));
    assert(std::fmax(s_nan, T(1)) == T(1));

    assert(std::isnan(std::fmax(nan, nan)));
  }
};

struct TestInt {
  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() const {
    using lim = std::numeric_limits<T>;
    TEST_CONSTEXPR_CXX23 T max = lim::max();
    TEST_CONSTEXPR_CXX23 T low = lim::lowest();

    assert(std::fmax(T(0), T(1)) == T(1));
    assert(std::fmax(T(1), T(0)) == T(1));
    assert(std::fmax(low, max) == max); 

    if (std::is_signed<T>::value) {
      assert(std::fmax(T(-1), T(0)) == T(0));
      assert(std::fmax(low, T(0)) == T(0));
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
