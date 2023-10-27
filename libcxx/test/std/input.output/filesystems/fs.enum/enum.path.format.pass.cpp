//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// class path;
// enum class format;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  typedef fs::path::format E;
  static_assert(std::is_enum<E>::value, "");

  typedef std::underlying_type<E>::type UT;

  LIBCPP_ONLY(static_assert(std::is_same<UT, unsigned char>::value, "")); // Implementation detail

  static_assert(
          E::auto_format   != E::native_format &&
          E::auto_format   != E::generic_format &&
          E::native_format != E::generic_format,
        "Expected enumeration values are not unique");

  return 0;
}
