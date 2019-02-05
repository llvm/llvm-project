//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <filesystem>

// enum class file_type;

#include "filesystem_include.hpp"
#include <type_traits>
#include <cassert>

#include "test_macros.h"


constexpr fs::file_type ME(int val) { return static_cast<fs::file_type>(val); }

int main(int, char**) {
  typedef fs::file_type E;
  static_assert(std::is_enum<E>::value, "");

  // Check that E is a scoped enum by checking for conversions.
  typedef std::underlying_type<E>::type UT;
  static_assert(!std::is_convertible<E, UT>::value, "");

  static_assert(std::is_same<UT, signed char>::value, ""); // Implementation detail

  static_assert(
          E::none == ME(0) &&
          E::not_found == ME(-1) &&
          E::regular == ME(1) &&
          E::directory == ME(2) &&
          E::symlink == ME(3) &&
          E::block == ME(4) &&
          E::character == ME(5) &&
          E::fifo == ME(6) &&
          E::socket == ME(7) &&
          E::unknown == ME(8),
        "Expected enumeration values do not match");

  return 0;
}
