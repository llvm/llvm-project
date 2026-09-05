//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool copysign(floating-point-type x, floating-point-type y); // constexpr since C++23

// We don't control the implementation on windows
// UNSUPPORTED: windows

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

    assert(std::isnan(std::copysign(lim::quiet_NaN(), T(-1))));
    assert(std::signbit(std::copysign(lim::signaling_NaN(), T(-1))));

    assert(std::copysign(lim::infinity(), -lim::infinity()) == -lim::infinity());
    assert(std::copysign(-lim::infinity(), lim::infinity()) == lim::infinity());

    assert(std::copysign(T(1), T(-1)) == T(-1));
    assert(std::copysign(T(-1), T(1)) == T(1));

    assert(std::copysign(lim::max(), -lim::max()) == -lim::max());
    assert(std::copysign(-lim::max(), lim::max()) == lim::max());

    assert(std::copysign(lim::denorm_min(), T(-1)) == -lim::denorm_min());
    assert(std::copysign(-lim::denorm_min(), T(1)) == lim::denorm_min());

    assert(std::copysign(T(0), T(-0.0)) == T(0));
    assert(std::copysign(T(-0.0), T(0)) == T(-0.0));
  }
};

struct TestInt {
  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() const {
    using lim = std::numeric_limits<T>;

    // no negative-zero bit pattern.
    assert(std::copysign(lim::max(), T(-0)) == static_cast<double>(lim::max()));
    assert(std::copysign(T(-0), lim::max()) == T(0));

    if (lim::is_signed) {
      assert(std::copysign(T(1), T(-1)) == T(-1));
      assert(std::copysign(T(-1), T(1)) == T(1));

      assert(std::copysign(T(1), lim::lowest()) == T(-1));
      assert(std::copysign(lim::lowest(), T(-1)) == lim::lowest());
    }
  }
};

TEST_CONSTEXPR_CXX23 bool test() {
  using lim = std::numeric_limits<double>;

  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::integral_types(), TestInt());

  // Make sure we can call `std::copysign` with mixed-type promotions.
  {
    assert(std::copysign(2.0, -1) == -2.0);                  // double vs int
    assert(std::copysign(-1, 2.0f) == 1);                    // int vs float
    assert(std::copysign(2.0L, -1.0f) == -2.0L);             // long double vs float
    assert(std::isnan(std::copysign(lim::quiet_NaN(), -1))); // NaN vs int
  }

  // std::copysignf(float, float).
  {
    assert(std::copysignf(2.0f, -1.0f) == -2.0f);
    assert(std::copysignf(-2.0f, 1.0f) == 2.0f);
  }

  // std::copysignl(long double, long double).
  {
    assert(std::copysignl(2.0L, -1.0L) == -2.0L);
    assert(std::copysignl(-2.0L, 1.0L) == 2.0L);
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
