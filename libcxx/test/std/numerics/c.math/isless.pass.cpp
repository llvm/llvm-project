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
    using lim = std::numeric_limits<T>;

    assert(!std::isless(lim::max(), T(0)));
    assert(std::isless(T(0), lim::max()));

    assert(!std::isless(lim::infinity(), lim::max()));
    assert(std::isless(-lim::infinity(), lim::lowest()));

    assert(!std::isless(T(0), lim::quiet_NaN()));
  }
};

struct TestInt {
  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() const {
    using lim = std::numeric_limits<T>;

    assert(!std::isless(lim::max(), T(0)));
    assert(std::isless(T(0), lim::max()));
    assert(!std::isless(lim::max(), lim::max()));

    assert(!std::isless(T(1), T(1)));

    if (lim::is_signed) {
      assert(!std::isless(T(-1), lim::lowest()));
      assert(std::isless(lim::lowest(), T(-1)));
    }
  }
};

TEST_CONSTEXPR_CXX23 bool test() {
  using lim = std::numeric_limits<double>;

  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::integral_types(), TestInt());

  // Make sure we can call `std::isless` with mixed-type promotions.
  {
    assert(!std::isless(2.0, 1));                // double vs int
    assert(std::isless(1, 2.0f));                // int vs float
    assert(!std::isless(2.0L, 1.0f));            // long double vs float
    assert(!std::isless(lim::quiet_NaN(), 1.0)); // NaN vs int
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
