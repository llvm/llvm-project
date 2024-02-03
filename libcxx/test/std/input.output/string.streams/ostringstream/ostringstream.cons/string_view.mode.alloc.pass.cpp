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
//   basic_ostringstream(const T& t, ios_base::openmode which, const Allocator& a);

#include <array>
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
  static_assert(test_convertible<std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                 const CharT*,
                                 std::ios_base::openmode,
                                 test_allocator<CharT>>());
  // `std::basic_string_view<CharT>`
  static_assert(test_convertible<std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                 std::basic_string_view<CharT>,
                                 std::ios_base::openmode,
                                 test_allocator<CharT>>());
  // `std::basic_string<CharT>`
  static_assert(test_convertible<std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                 std::basic_string<CharT>,
                                 std::ios_base::openmode,
                                 test_allocator<CharT>>());

  // Non-convertible to std::`basic_string_view<CharT>`
  static_assert(!test_convertible<std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  int,
                                  std::ios_base::openmode,
                                  test_allocator<CharT>>());
  static_assert(!test_convertible<std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  const int&,
                                  std::ios_base::openmode,
                                  test_allocator<CharT>>());
  static_assert(!test_convertible<std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  SomeObject,
                                  std::ios_base::openmode,
                                  test_allocator<CharT>>());
  static_assert(!test_convertible<std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  const SomeObject&,
                                  std::ios_base::openmode,
                                  test_allocator<CharT>>());

  // Mode
  static_assert(!test_convertible<std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  std::basic_string<CharT>,
                                  SomeObject,
                                  NonAllocator>());

  // Allocator
  static_assert(!test_convertible<std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  std::basic_string<CharT>,
                                  std::ios_base::openmode,
                                  NonAllocator>());
}

#define CS(S) MAKE_CSTRING(CharT, S)
#define ST(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static void test() {
  using namespace std::string_literals;
  using namespace std::string_view_literals;

  // const CharT*
  {
    const std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(CS("zmt"));
    assert(ss.view() == SV("zmt"));
  }
  {
    const std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(
        CS("zmt"), std::ios_base::binary);
    assert(ss.view() == SV("zmt"));
  }
  // std::basic_string_view<CharT>
  {
    const std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(SV("zmt"));
    assert(ss.view() == SV("zmt"));
  }
  {
    const std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(
        SV("zmt"), std::ios_base::binary);
    assert(ss.view() == SV("zmt"));
  }
  // std::basic_string<CharT>
  {
    const std::basic_string<CharT> s(ST("zmt"));
    const std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(s);
    assert(ss.view() == SV("zmt"));
  }
  {
    const std::basic_string<CharT> s(ST("zmt"));
    const std::basic_ostringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(s, std::ios_base::binary);
    assert(ss.view() == SV("zmt"));
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
