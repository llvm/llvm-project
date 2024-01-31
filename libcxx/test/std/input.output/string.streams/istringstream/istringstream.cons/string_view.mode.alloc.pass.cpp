//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT>>
// class basic_stringstream

// template<class T>
//   basic_istringstream(const T& t, ios_base::openmode which, const Allocator& a);

#include <array>
#include <cassert>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

#include "make_string.h"
#include "test_allocator.h"
#include "test_macros.h"

using namespace std::string_literals;
using namespace std::string_view_literals;

#define STR(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static void test() {
  // const ChartT*
  {
    const std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(STR("zmt"));
    assert(ss.view() == SV("zmt"));
  }
  {
    const std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(
        STR("zmt"), std::ios_base::binary);
    assert(ss.view() == SV("zmt"));
  }
  // basic_string_view<CharT>
  {
    const std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(SV("zmt"));
    assert(ss.view() == SV("zmt"));
  }
  {
    const std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(
        SV("zmt"), std::ios_base::binary);
    assert(ss.view() == SV("zmt"));
  }
  // basic_string<CharT>
  {
    const std::basic_string<CharT> s(STR("zmt"));
    const std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(s);
    assert(ss.view() == SV("zmt"));
  }
  {
    const std::basic_string<CharT> s(STR("zmt"));
    const std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(s, std::ios_base::binary);
    assert(ss.view() == SV("zmt"));
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  return 0;
}