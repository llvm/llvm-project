//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <fstream>

// class basic_filebuf;

// native_handle_type native_handle() const noexcept;

#include <cassert>
#include <fstream>
#include <filesystem>
#include <utility>

#include "platform_support.h"
#include "test_macros.h"
#include "../test_helpers.h"

template <typename CharT>
void test() {
  std::basic_filebuf<CharT> f;
  assert(!f.is_open());
  std::filesystem::path p = get_temp_file_name();
  f.open(p, std::ios_base::in);
  assert(f.is_open());
  std::same_as<HandleT> decltype(auto) handle = f.native_handle();
  assert(is_handle_valid(handle));
  std::same_as<HandleT> decltype(auto) const_handle = std::as_const(f).native_handle();
  assert(is_handle_valid(const_handle));
  static_assert(noexcept(f.native_handle()));
  static_assert(noexcept(std::as_const(f).native_handle()));
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
