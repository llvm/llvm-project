//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// REQUIRES: has-unix-headers
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: availability-verbose_abort-missing

// <fstream>

// class basic_filebuf;

// native_handle_type native_handle() const noexcept;

#include <fstream>

#include "../native_handle_assert_test_helpers.h"

int main(int, char**) {
  test_native_handle_assertion<std::basic_filebuf<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_native_handle_assertion<std::basic_filebuf<wchar_t>>();
#endif

  return 0;
}
