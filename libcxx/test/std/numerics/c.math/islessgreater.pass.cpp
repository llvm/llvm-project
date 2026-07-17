//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool islessgreater(floating-point-type x, floating-point-type y); // constexpr since C++23

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
    assert(std::islessgreater(T(1), T(2)));
    assert(std::islessgreater(T(2), T(1)));
    assert(!std::islessgreater(T(1), T(1)));

    assert(!std::islessgreater(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity()));
    assert(std::islessgreater(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::max()));

    assert(!std::islessgreater(std::numeric_limits<T>::quiet_NaN(), T(0)));
    assert(!std::islessgreater(T(0), std::numeric_limits<T>::quiet_NaN()));
    assert(!std::islessgreater(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()));

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
    if (!std::is_same<T, bool>::value) {
      assert(std::islessgreater(T(1), T(2)));
      assert(std::islessgreater(T(2), T(1)));
    }

    assert(!std::islessgreater(T(1), T(1)));

    assert(!std::islessgreater(T(1), T(1)));
    assert(std::islessgreater(std::numeric_limits<T>::max(), T(0)));
    assert(!std::islessgreater(std::numeric_limits<T>::max(), std::numeric_limits<T>::max()));

    if (std::is_signed<T>::value) {
      assert(std::islessgreater(T(-1), T(1)));
      assert(std::islessgreater(std::numeric_limits<T>::lowest(), T(0)));
      assert(std::islessgreater(T(0), std::numeric_limits<T>::lowest()));
      assert(!std::islessgreater(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest()));
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

  return 0;
}
