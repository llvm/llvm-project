//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool islessgreater(floating-point-type x, floating-point-type y); // constexpr since C++23

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

#include "test_macros.h"
#include "type_algorithms.h"

struct TestFloat {
  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() const {
    using lim                  = std::numeric_limits<T>;
    TEST_CONSTEXPR_CXX23 T max = lim::max();
    TEST_CONSTEXPR_CXX23 T inf = lim::infinity();
    TEST_CONSTEXPR_CXX23 T nan = lim::quiet_NaN();

    assert(std::islessgreater(T(1), T(2)));
    assert(std::islessgreater(T(2), T(1)));
    assert(!std::islessgreater(T(1), T(1)));

    assert(!std::islessgreater(inf, inf));
    assert(std::islessgreater(inf, max));

    assert(!std::islessgreater(nan, T(0)));
    assert(!std::islessgreater(T(0), nan));
    assert(!std::islessgreater(nan, nan));
  }
};

struct TestInt {
  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() const {
    using lim                  = std::numeric_limits<T>;
    TEST_CONSTEXPR_CXX23 T max = lim::max();
    TEST_CONSTEXPR_CXX23 T low = lim::lowest();

    if (!std::is_same<T, bool>::value) {
      assert(std::islessgreater(T(1), T(2)));
      assert(std::islessgreater(T(2), T(1)));
    }

    assert(!std::islessgreater(T(1), T(1)));
    assert(std::islessgreater(max, T(0)));
    assert(!std::islessgreater(max, max));

    if (lim::is_signed) {
      assert(std::islessgreater(T(-1), T(1)));
      assert(std::islessgreater(low, T(0)));
      assert(std::islessgreater(T(0), low));
      assert(!std::islessgreater(low, low));
    }
  }
};

TEST_CONSTEXPR_CXX23 bool test() {
  using lim                     = std::numeric_limits<double>;
  TEST_CONSTEXPR_CXX23 auto nan = lim::quiet_NaN();

  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::integral_types(), TestInt());

  // Make sure we can call `std::islessgreater` with mixed-type promotions with __promote_t<_A1, _A2>.
  {
    assert(std::islessgreater(2.0, 1));     // double vs int
    assert(std::islessgreater(1, 2.0f));    // int vs float
    assert(std::islessgreater(2.0L, 1.0f)); // long double vs float
    assert(!std::islessgreater(nan, 1.0));  // NaN vs int
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
