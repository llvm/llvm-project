//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <fstream>

// class basic_fstream;

// native_handle_type native_handle() const noexcept;

#include "test_macros.h"
#include "../native_handle_test_helpers.h"

int main(int, char**) {
  test_native_handle<char, std::basic_fstream<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_native_handle<wchar_t, std::basic_fstream<wchar_t>>();
#endif

  return 0;
}
