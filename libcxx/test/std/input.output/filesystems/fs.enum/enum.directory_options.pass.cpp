//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <filesystem>

// enum class directory_options;

#include "filesystem_include.hpp"
#include <type_traits>
#include <cassert>
#include <sys/stat.h>

#include "test_macros.h"
#include "check_bitmask_types.hpp"


constexpr fs::directory_options ME(int val) { return static_cast<fs::directory_options>(val); }

int main() {
  typedef fs::directory_options E;
  static_assert(std::is_enum<E>::value, "");

  // Check that E is a scoped enum by checking for conversions.
  typedef std::underlying_type<E>::type UT;
  static_assert(!std::is_convertible<E, UT>::value, "");
  static_assert(std::is_same<UT, unsigned char>::value, "");

  typedef check_bitmask_type<E, E::follow_directory_symlink, E::skip_permission_denied> BitmaskTester;
  assert(BitmaskTester::check());

  static_assert(
        E::none                     == ME(0) &&
        E::follow_directory_symlink == ME(1) &&
        E::skip_permission_denied   == ME(2),
        "Expected enumeration values do not match");

}
