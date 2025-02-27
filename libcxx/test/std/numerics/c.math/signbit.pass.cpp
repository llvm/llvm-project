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

// These compilers don't support constexpr `__builtin_signbit` yet.
// UNSUPPORTED: clang-18, clang-19, apple-clang-15, apple-clang-16, apple-clang-17

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <cassert>
#include <cmath>
#include <limits>

#include "test_macros.h"
#include "type_algorithms.h"

struct TestFloat {
  template <class T>
  static TEST_CONSTEXPR_CXX23 bool test() {
    assert(!std::signbit(T(0)));
    assert(!std::signbit(std::numeric_limits<T>::min()));
    assert(!std::signbit(std::numeric_limits<T>::denorm_min()));
    assert(!std::signbit(std::numeric_limits<T>::max()));
    assert(!std::signbit(std::numeric_limits<T>::infinity()));
    assert(!std::signbit(std::numeric_limits<T>::quiet_NaN()));
    assert(!std::signbit(std::numeric_limits<T>::signaling_NaN()));
    assert(std::signbit(-T(0)));
    assert(std::signbit(-std::numeric_limits<T>::infinity()));
    assert(std::signbit(std::numeric_limits<T>::lowest()));

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
    assert(!std::signbit(std::numeric_limits<T>::max()));
    assert(!std::signbit(T(0)));
    if (std::is_unsigned<T>::value) {
      assert(!std::signbit(std::numeric_limits<T>::lowest()));
    } else {
      assert(std::signbit(std::numeric_limits<T>::lowest()));
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
  operator T() const { return T(); }
};

int main(int, char**) {
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
  return 0;
}
