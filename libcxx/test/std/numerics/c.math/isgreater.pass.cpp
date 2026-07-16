//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool isgreater(floating-point-type x, floating-point-type y); // constexpr since C++23

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
    assert(std::isgreater(std::numeric_limits<T>::max(), T(0)));

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
    assert(std::isgreater(std::numeric_limits<T>::max(), T(0)));

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

template <typename T>
struct ZeroConvertibleTo {
  operator T() const { return T(0); }
};

int main(int, char**) {
  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::integral_types(), TestInt());

  return 0;
}
