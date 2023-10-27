//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_NODISCARD_EXT

// Check that <bit> functions aren't marked [[nodiscard]] when
// _LIBCPP_DISBALE_NODISCARD_EXT is defined

#include <bit>

#include "test_macros.h"

void func() {
  std::bit_cast<unsigned int>(42);
  std::bit_ceil(0u);
  std::bit_floor(0u);
  std::bit_width(0u);
#if TEST_STD_VER >= 23
  std::byteswap(0u);
#endif
  std::countl_zero(0u);
  std::countl_one(0u);
  std::countr_zero(0u);
  std::countr_one(0u);
  std::has_single_bit(0u);
  std::popcount(0u);
}
