//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_NODISCARD_EXT

// Check that <limits> functions aren't marked [[nodiscard]] when
// _LIBCPP_DISABLE_NODISCARD_EXT is defined

#include <limits>

#include "test_macros.h"

void func() {
  // arithmetic
  std::numeric_limits<int>::min();
  std::numeric_limits<int>::max();
  std::numeric_limits<int>::lowest();
  std::numeric_limits<int>::epsilon();
  std::numeric_limits<int>::round_error();
  std::numeric_limits<int>::infinity();
  std::numeric_limits<int>::quiet_NaN();
  std::numeric_limits<int>::signaling_NaN();
  std::numeric_limits<int>::denorm_min();
  // bool
  std::numeric_limits<bool>::min();
  std::numeric_limits<bool>::max();
  std::numeric_limits<bool>::lowest();
  std::numeric_limits<bool>::epsilon();
  std::numeric_limits<bool>::round_error();
  std::numeric_limits<bool>::infinity();
  std::numeric_limits<bool>::quiet_NaN();
  std::numeric_limits<bool>::signaling_NaN();
  std::numeric_limits<bool>::denorm_min();
  // float
  std::numeric_limits<float>::min();
  std::numeric_limits<float>::max();
  std::numeric_limits<float>::lowest();
  std::numeric_limits<float>::epsilon();
  std::numeric_limits<float>::round_error();
  std::numeric_limits<float>::infinity();
  std::numeric_limits<float>::quiet_NaN();
  std::numeric_limits<float>::signaling_NaN();
  std::numeric_limits<float>::denorm_min();
  // double
  std::numeric_limits<double>::min();
  std::numeric_limits<double>::max();
  std::numeric_limits<double>::lowest();
  std::numeric_limits<double>::epsilon();
  std::numeric_limits<double>::round_error();
  std::numeric_limits<double>::infinity();
  std::numeric_limits<double>::quiet_NaN();
  std::numeric_limits<double>::signaling_NaN();
  std::numeric_limits<double>::denorm_min();
  // long double
  std::numeric_limits<long double>::min();
  std::numeric_limits<long double>::max();
  std::numeric_limits<long double>::lowest();
  std::numeric_limits<long double>::epsilon();
  std::numeric_limits<long double>::round_error();
  std::numeric_limits<long double>::infinity();
  std::numeric_limits<long double>::quiet_NaN();
  std::numeric_limits<long double>::signaling_NaN();
  std::numeric_limits<long double>::denorm_min();
}
