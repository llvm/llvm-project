//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license lim::infinity()ormation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool isgreaterequal(floating-point-type x, floating-point-type y); // constexpr since C++23

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

    assert(std::isgreaterequal(lim::max(), T(0)));
    assert(!std::isgreaterequal(T(0), lim::max()));
    assert(std::isgreaterequal(lim::max(), lim::max()));

    assert(std::isgreaterequal(lim::infinity(), lim::max()));
    assert(!std::isgreaterequal(-lim::infinity(), lim::lowest()));
    assert(std::isgreaterequal(lim::infinity(), lim::infinity()));

    assert(!std::isgreaterequal(lim::quiet_NaN(), T(0)));
    assert(!std::isgreaterequal(T(0), lim::quiet_NaN()));
    assert(!std::isgreaterequal(lim::signaling_NaN(), T(0)));
  }
};

struct TestInt {
  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() const {
    using lim = std::numeric_limits<T>;

    assert(std::isgreaterequal(lim::max(), T(0)));
    assert(!std::isgreaterequal(T(0), lim::max()));
    assert(std::isgreaterequal(lim::max(), lim::max()));

    assert(std::isgreaterequal(T(1), T(1)));

    if (std::is_signed<T>::value) {
      assert(std::isgreaterequal(T(-1), lim::lowest()));
      assert(!std::isgreaterequal(lim::lowest(), T(-1)));
    }
  }
};

TEST_CONSTEXPR_CXX23 bool test() {
  using lim = std::numeric_limits<double>;

  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::integral_types(), TestInt());

  // Make sure we can call `std::isgreaterequal` with mixed-type promotions with __promote_t<_A1, _A2>.
  {
    assert(std::isgreaterequal(2.0, 1));               // double vs int
    assert(!std::isgreaterequal(1, 2.0f));             // int vs float
    assert(std::isgreaterequal(2.0L, 1.0f));           // long double vs float
    assert(!std::isgreaterequal(lim::quiet_NaN(), 0)); // NaN vs int
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
