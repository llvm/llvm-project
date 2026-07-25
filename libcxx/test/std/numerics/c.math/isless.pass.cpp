//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool isless(floating-point-type x, floating-point-type y); // constexpr since C++23

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
    TEST_CONSTEXPR_CXX23 T low   = lim::lowest();
    TEST_CONSTEXPR_CXX23 T inf   = lim::infinity();
    TEST_CONSTEXPR_CXX23 T nan   = lim::quiet_NaN();

    assert(!std::isless(max, T(0)));
    assert(std::isless(T(0), max));

    assert(!std::isless(inf, max));
    assert(std::isless(-inf, low));

    assert(!std::isless(T(0), nan));
  }
};

struct TestInt {
  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() const {
    using lim                  = std::numeric_limits<T>;
    TEST_CONSTEXPR_CXX23 T max = lim::max();
    TEST_CONSTEXPR_CXX23 T low = lim::lowest();

    assert(!std::isless(max, T(0)));
    assert(std::isless(T(0), max));
    assert(!std::isless(max, max));

    assert(!std::isless(T(1), T(1)));

    if (lim::is_signed) {
      assert(!std::isless(T(-1), low));
      assert(std::isless(low, T(-1)));
    }
  }
};

TEST_CONSTEXPR_CXX23 bool test() {
  using lim                     = std::numeric_limits<double>;
  TEST_CONSTEXPR_CXX23 auto nan = lim::quiet_NaN();

  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::integral_types(), TestInt());

  // Make sure we can call `std::isless` with mixed-type promotions with __promote_t<_A1, _A2>.
  {
    assert(!std::isless(2.0, 1));     // double vs int
    assert(std::isless(1, 2.0f));     // int vs float
    assert(!std::isless(2.0L, 1.0f)); // long double vs float
    assert(!std::isless(nan, 1.0));   // NaN vs int
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif
  return 0;
}
