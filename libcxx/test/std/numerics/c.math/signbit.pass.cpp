//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool signbit(floating-point-type x); // constexpr since C++23

// We don't control the implementation on windows
// UNSUPPORTED: windows

// GCC warns about signbit comparing `bool_v < 0`, which we're testing
// ADDITIONAL_COMPILE_FLAGS(gcc): -Wno-bool-compare

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

#include "test_macros.h"
#include "type_algorithms.h"

struct TestFloat {
  template <class T>
  static TEST_CONSTEXPR_CXX23 void operator()() {
    using lim = std::numeric_limits<T>;

    assert(!std::signbit(T(0)));
    assert(!std::signbit(lim::min()));
    assert(!std::signbit(lim::denorm_min()));
    assert(!std::signbit(lim::max()));
    assert(!std::signbit(lim::infinity()));
    assert(!std::signbit(lim::quiet_NaN()));
    assert(!std::signbit(lim::signaling_NaN()));
    assert(std::signbit(-T(0)));
    assert(std::signbit(-lim::infinity()));
    assert(std::signbit(lim::lowest()));
  }
};

struct TestInt {
  template <class T>
  static TEST_CONSTEXPR_CXX23 void operator()() {
    using lim = std::numeric_limits<T>;

    assert(!std::signbit(lim::max()));
    assert(!std::signbit(T(0)));
    if (std::is_unsigned<T>::value) {
      assert(!std::signbit(lim::lowest()));
    } else {
      assert(std::signbit(lim::lowest()));
    }
  }
};

template <typename T>
struct ConvertibleTo {
  TEST_CONSTEXPR_CXX23 operator T() const { return T(); }
};

TEST_CONSTEXPR_CXX23 bool test() {
  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::integral_types(), TestInt());

  // Make sure we can call `std::signbit` with convertible types. This checks
  // whether overloads for all cv-unqualified floating-point types are working
  // as expected.
  {
    assert(!std::signbit(ConvertibleTo<float>()));
    assert(!std::signbit(ConvertibleTo<double>()));
    assert(!std::signbit(ConvertibleTo<long double>()));
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
