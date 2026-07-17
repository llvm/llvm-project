//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool islessequalequal(floating-point-type x, floating-point-type y); // constexpr since C++23

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
  static TEST_CONSTEXPR_CXX23 bool test() {
    assert(!std::islessequal(std::numeric_limits<T>::max(), T(0)));
    assert(std::islessequal(T(0), std::numeric_limits<T>::max()));
    assert(std::islessequal(std::numeric_limits<T>::max(), std::numeric_limits<T>::max()));

    assert(!std::islessequal(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::max()));
    assert(std::islessequal(-std::numeric_limits<T>::infinity(), std::numeric_limits<T>::lowest()));
    assert(std::islessequal(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity()));

    assert(!std::islessequal(std::numeric_limits<T>::quiet_NaN(), T(0)));
    assert(!std::islessequal(T(0), std::numeric_limits<T>::quiet_NaN()));
    assert(!std::islessequal(std::numeric_limits<T>::signaling_NaN(), T(0)));

    return true;
  }

  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() {
    test<T>();
#if TEST_STD_VER >= 23
    static_assert(test<T>());
#endif
  }
};

struct TestInt {
  template <class T>
  static TEST_CONSTEXPR_CXX23 bool test() {
    assert(!std::islessequal(std::numeric_limits<T>::max(), T(0)));
    assert(std::islessequal(T(0), std::numeric_limits<T>::max()));
    assert(std::islessequal(std::numeric_limits<T>::max(), std::numeric_limits<T>::max()));

    assert(std::islessequal(T(1), T(1)));

    if (std::is_signed<T>::value) {
      assert(!std::islessequal(T(-1), std::numeric_limits<T>::lowest()));
      assert(std::islessequal(std::numeric_limits<T>::lowest(), T(-1)));
    }

    return true;
  }

  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() {
    test<T>();
#if TEST_STD_VER >= 23
    static_assert(test<T>());
#endif
  }
};

template <typename T>
struct ConvertibleTo {
  operator T() const { return T(1); }
};

int main(int, char**) {
  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::integral_types(), TestInt());

  // Make sure we can call `std::islessequal` with mixed-type promotions with __promote_t<_A1, _A2>.
  {
    assert(!std::islessequal(2.0, 1));     // double vs int
    assert(std::islessequal(1, 2.0f));   // int vs float
    assert(!std::islessequal(2.0L, 1.0f)); // long double vs float
  }

  return 0;
}
