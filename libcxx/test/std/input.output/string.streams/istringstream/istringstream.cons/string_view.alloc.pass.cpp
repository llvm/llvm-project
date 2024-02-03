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
//   basic_istringstream(const T& t, const Allocator& a);

#include <cassert>
#include <sstream>
#include <string>
#include <string_view>

#include "make_string.h"
#include "test_allocator.h"
#include "test_convertible.h"
#include "test_macros.h"

struct SomeObject {};
struct NonAllocator {};

template <typename CharT>
void test_constraints() {
  // `const CharT*`
  static_assert(test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                 const CharT*,
                                 test_allocator<CharT>>());
  // `std::basic_string_view<CharT>`
  static_assert(test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                 std::basic_string_view<CharT>,
                                 test_allocator<CharT>>());
  // `std::basic_string<CharT>`
  static_assert(test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                 std::basic_string<CharT>,
                                 test_allocator<CharT>>());

  // Non-convertible to std::`basic_string_view<CharT>`
  static_assert(!test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  int,
                                  test_allocator<CharT>>());
  static_assert(!test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  const int&,
                                  test_allocator<CharT>>());
  static_assert(!test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  SomeObject,

                                  test_allocator<CharT>>());
  static_assert(!test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  const SomeObject&,
                                  test_allocator<CharT>>());

  // Allocator
  static_assert(!test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  std::basic_string<CharT>,
                                  NonAllocator>());
}

#define CS(S) MAKE_CSTRING(CharT, S)
#define ST(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static void test() {
  const test_allocator<CharT> ca;

  // const CharT*
  {
    const std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(CS("zmt"), ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
  // std::basic_string_view<CharT>
  {
    const auto csv = SV("zmt");
    const std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(csv, ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
  // std::basic_string<CharT>
  {
    const auto cs = ST("zmt");
    const std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(cs, ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
}

int main(int, char**) {
  test_constraints<char>();
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_constraints<wchar_t>();
  test<wchar_t>();
#endif
  return 0;
}
