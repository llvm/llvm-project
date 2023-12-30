//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <fstream>

// class basic_ifstream;

// native_handle_type native_handle() const noexcept;

#include <cassert>
#include <fstream>
#include <filesystem>
#include <utility>

#if defined(_LIBCPP_WIN32API)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <io.h>
#  include <windows.h>
#endif

#include "platform_support.h"
#include "test_macros.h"
#include "../test_helpers.h"

template <typename CharT>
void test() {
  static_assert(std::is_same_v<typename std::basic_filebuf<CharT>::native_handle_type,
                               typename std::basic_ifstream<CharT>::native_handle_type>);

  std::basic_ifstream<CharT> f;
  static_assert(noexcept(f.native_handle()));
  assert(!f.is_open());
  std::filesystem::path p = get_temp_file_name();
  f.open(p);
  assert(f.is_open());
  assert(f.native_handle() == f.rdbuf()->native_handle());
  assert(is_handle_valid(f.native_handle()));
  assert(is_handle_valid(std::as_const(f).native_handle()));
  static_assert(noexcept(f.native_handle()));
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
