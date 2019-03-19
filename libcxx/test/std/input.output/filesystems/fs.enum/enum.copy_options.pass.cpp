//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <filesystem>

// enum class copy_options;

#include "filesystem_include.hpp"
#include <type_traits>
#include <cassert>

#include "check_bitmask_types.hpp"
#include "test_macros.h"


constexpr fs::copy_options ME(int val) { return static_cast<fs::copy_options>(val); }

int main(int, char**) {
  typedef fs::copy_options E;
  static_assert(std::is_enum<E>::value, "");

  // Check that E is a scoped enum by checking for conversions.
  typedef std::underlying_type<E>::type UT;
  static_assert(!std::is_convertible<E, UT>::value, "");

  static_assert(std::is_same<UT, unsigned short>::value, ""); // Implementation detail

  typedef check_bitmask_type<E, E::skip_existing, E::update_existing> BitmaskTester;
  assert(BitmaskTester::check());

  static_assert(
          E::none == ME(0),
        "Expected enumeration values do not match");
  // Option group for copy_file
  static_assert(
          E::skip_existing      == ME(1) &&
          E::overwrite_existing == ME(2) &&
          E::update_existing    == ME(4),
        "Expected enumeration values do not match");
  // Option group for copy on directories
  static_assert(
          E::recursive == ME(8),
        "Expected enumeration values do not match");
  // Option group for copy on symlinks
  static_assert(
          E::copy_symlinks == ME(16) &&
          E::skip_symlinks == ME(32),
        "Expected enumeration values do not match");
  // Option group for changing form of copy
  static_assert(
          E::directories_only    == ME(64) &&
          E::create_symlinks     == ME(128) &&
          E::create_hard_links   == ME(256),
        "Expected enumeration values do not match");

  return 0;
}
