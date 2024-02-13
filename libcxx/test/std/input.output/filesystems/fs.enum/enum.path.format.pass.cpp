//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// class path;
// enum class format;

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
namespace fs = std::filesystem;

int main(int, char**) {
  typedef fs::path::format E;
  static_assert(std::is_enum<E>::value, "");

  LIBCPP_STATIC_ASSERT(std::is_same<std::underlying_type<E>::type, unsigned char>::value, ""); // Implementation detail

  static_assert(
          E::auto_format   != E::native_format &&
          E::auto_format   != E::generic_format &&
          E::native_format != E::generic_format,
        "Expected enumeration values are not unique");

  return 0;
}
