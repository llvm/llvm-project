//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool isgreater(floating-point-type x, floating-point-type y); // constexpr since C++23

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

    assert(std::isgreater(lim::max(), T(0)));
    assert(!std::isgreater(T(0), lim::max()));
    assert(!std::isgreater(lim::max(), lim::max()));

    assert(std::isgreater(lim::infinity(), lim::max()));
    assert(!std::isgreater(-lim::infinity(), lim::lowest()));
    assert(!std::isgreater(lim::infinity(), lim::infinity()));

    assert(!std::isgreater(lim::quiet_NaN(), T(0)));
    assert(!std::isgreater(T(0), lim::quiet_NaN()));
    assert(!std::isgreater(lim::quiet_NaN(), lim::quiet_NaN()));
    assert(!std::isgreater(lim::signaling_NaN(), T(0)));
  }
};

struct TestInt {
  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() const {
    using lim = std::numeric_limits<T>;

    assert(std::isgreater(lim::max(), T(0)));
    assert(!std::isgreater(T(0), lim::max()));
    assert(!std::isgreater(lim::max(), lim::max()));

    assert(!std::isgreater(T(1), T(1)));
    assert(!std::isgreater(lim::lowest(), T(0)));

    if (lim::is_signed) {
      assert(std::isgreater(T(-1), lim::lowest()));
      assert(!std::isgreater(lim::lowest(), T(-1)));
    }
  }
};

TEST_CONSTEXPR_CXX23 bool test() {
  using lim = std::numeric_limits<double>;

  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::integral_types(), TestInt());

  {
    assert(std::isgreater(2.0, 1));               // double vs int
    assert(!std::isgreater(1, 2.0f));             // int vs float
    assert(std::isgreater(2.0L, 1.0f));           // long double vs float
    assert(!std::isgreater(lim::quiet_NaN(), 0)); // NaN vs int
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
