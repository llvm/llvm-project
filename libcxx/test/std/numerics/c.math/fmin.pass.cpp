//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// floating-point-type fmin(floating-point-type x, floating-point-type y); // constexpr since C++23

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

    assert(std::fmin(T(1), T(2)) == T(1));
    assert(std::fmin(T(2), T(1)) == T(1));
    assert(std::fmin(T(-1), T(0)) == T(-1));

    assert(std::fmin(inf, T(1)) == T(1));
    assert(std::fmin(T(1), inf) == T(1));
    assert(std::fmin(-inf, T(1)) == -inf);
    assert(std::fmin(T(1), -inf) == -inf);

    assert(std::fmin(nan, T(1)) == T(1));
    assert(std::fmin(T(1), nan) == T(1));

    assert(std::isnan(std::fmin(nan, nan)));
  }
};

struct TestInt {
  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() const {
    using lim = std::numeric_limits<T>;

    assert(std::fmin(T(0), T(1)) == T(0));
    assert(std::fmin(T(1), T(0)) == T(0));

    if (std::is_signed<T>::value) {
      assert(std::fmin(T(-1), T(0)) == T(-1));
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
