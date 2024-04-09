// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_CHRONO_LEAP_SECOND_HPP
#define SUPPORT_TEST_CHRONO_LEAP_SECOND_HPP

// Contains helper functions to create a std::chrono::leap_second.
//
// Since the standard doesn't specify how a @ref std::chrono::leap_second is
// constructed this is implementation defined. To make the public API tests of
// the class generic this header defines helper functions to create the
// required object.
//
// Note This requires every standard library implementation to write their own
// helper function. Vendors are encouraged to create a pull request at
// https://github.com/llvm/llvm-project so their specific implementation can be
// part of this file.

#include "test_macros.h"

#if TEST_STD_VER < 20
#  error "The format header requires at least C++20"
#endif

#include <chrono>

#ifdef _LIBCPP_VERSION

// In order to find this include the calling test needs to provide this path in
// the search path. Typically this looks like:
//   ADDITIONAL_COMPILE_FLAGS(stdlib=libc++): -I %{libcxx-dir}/src/include
// where the number of `../` sequences depends on the subdirectory level of the
// test.
#  include "tzdb/leap_second_private.h" // Header in the dylib

inline constexpr std::chrono::leap_second
test_leap_second_create(const std::chrono::sys_seconds& date, const std::chrono::seconds& value) {
  return std::chrono::leap_second{std::chrono::leap_second::__constructor_tag{}, date, value};
}

#else // _LIBCPP_VERSION
#  error                                                                                                               \
      "Please create a vendor specific version of the test typedef and file a PR at https://github.com/llvm/llvm-project"
#endif // _LIBCPP_VERSION

#endif // SUPPORT_TEST_CHRONO_LEAP_SECOND_HPP
