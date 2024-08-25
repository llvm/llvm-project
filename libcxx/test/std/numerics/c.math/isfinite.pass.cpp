//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool isfinite(floating-point-type x); // constexpr since C++23

// We don't control the implementation on windows
// UNSUPPORTED: windows

#include <cassert>
#include <cmath>
#include <limits>

#include "test_macros.h"
#include "type_algorithms.h"

#if TEST_STD_VER >= 23
#  define COMPILE_OR_RUNTIME_ASSERT(expr) static_assert(expr)
#else
#  define COMPILE_OR_RUNTIME_ASSERT(expr) assert(expr)
#endif

struct TestFloat {
  template <class T>
  static void test() {
    COMPILE_OR_RUNTIME_ASSERT(std::isfinite(std::numeric_limits<T>::max()));
    COMPILE_OR_RUNTIME_ASSERT(!std::isfinite(std::numeric_limits<T>::infinity()));
    COMPILE_OR_RUNTIME_ASSERT(std::isfinite(std::numeric_limits<T>::min()));
    COMPILE_OR_RUNTIME_ASSERT(std::isfinite(std::numeric_limits<T>::denorm_min()));
    COMPILE_OR_RUNTIME_ASSERT(std::isfinite(std::numeric_limits<T>::lowest()));
    COMPILE_OR_RUNTIME_ASSERT(!std::isfinite(-std::numeric_limits<T>::infinity()));
    COMPILE_OR_RUNTIME_ASSERT(std::isfinite(T(0)));
    COMPILE_OR_RUNTIME_ASSERT(!std::isfinite(std::numeric_limits<T>::quiet_NaN()));
    COMPILE_OR_RUNTIME_ASSERT(!std::isfinite(std::numeric_limits<T>::signaling_NaN()));
  }

  template <class T>
  void operator()() {
    test<T>();
  }
};

struct TestInt {
  template <class T>
  static void test() {
    COMPILE_OR_RUNTIME_ASSERT(std::isfinite(std::numeric_limits<T>::max()));
    COMPILE_OR_RUNTIME_ASSERT(std::isfinite(std::numeric_limits<T>::lowest()));
    COMPILE_OR_RUNTIME_ASSERT(std::isfinite(T(0)));
  }

  template <class T>
  void operator()() {
    test<T>();
  }
};

template <typename T>
struct ConvertibleTo {
  TEST_CONSTEXPR_CXX23 operator T() const { return T(); }
};

int main(int, char**) {
  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::integral_types(), TestInt());

  // Make sure we can call `std::isfinite` with convertible types
  {
    COMPILE_OR_RUNTIME_ASSERT(std::isfinite(ConvertibleTo<float>()));
    COMPILE_OR_RUNTIME_ASSERT(std::isfinite(ConvertibleTo<double>()));
    COMPILE_OR_RUNTIME_ASSERT(std::isfinite(ConvertibleTo<long double>()));
  }

  return 0;
}
