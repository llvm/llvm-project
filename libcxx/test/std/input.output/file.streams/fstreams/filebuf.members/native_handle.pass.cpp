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
#include "../native_handle_test_helpers.h"

template <typename CharT>
void test() {
  std::basic_filebuf<CharT> f;
  std::filesystem::path p = get_temp_file_name();

  // non-const
  {
    assert(f.open(p, std::ios_base::in) != nullptr);
    std::same_as<NativeHandleT> decltype(auto) handle = f.native_handle();
    assert(is_handle_valid(handle));
    f.close();
    assert(!is_handle_valid(handle));
    static_assert(noexcept(f.native_handle()));
  }
  // const
  {
    assert(f.open(p, std::ios_base::in) != nullptr);
    std::same_as<NativeHandleT> decltype(auto) const_handle = std::as_const(f).native_handle();
    assert(is_handle_valid(const_handle));
    f.close();
    assert(!is_handle_valid(const_handle));
    static_assert(noexcept(std::as_const(f).native_handle()));
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
