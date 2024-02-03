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

#include "make_string.h"
#include "test_allocator.h"
#include "test_convertible.h"
#include "test_macros.h"

template <typename S, typename T, typename A>
concept HasMember = requires(const T& sv, std::ios_base::openmode w, const A& a) {
  { S(sv, w, a) };
};

template <typename CharT>
void test_constraints() {
  struct SomeObject {};
  struct NonAllocator {};

  // `const CharT*`
  static_assert(HasMember<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                          const CharT*,
                          test_allocator<CharT>>);
  static_assert(test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                 const CharT*,
                                 std::ios_base::openmode,
                                 test_allocator<CharT>>());
  // `std::basic_string_view<CharT>`
  static_assert(HasMember<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                          std::basic_string_view<CharT>,
                          test_allocator<CharT>>);
  static_assert(test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                 std::basic_string_view<CharT>,
                                 std::ios_base::openmode,
                                 test_allocator<CharT>>());
  // `std::basic_string<CharT>`
  static_assert(HasMember<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                          std::basic_string<CharT>,
                          std::ios_base::openmode,
                          test_allocator<CharT>>);
  static_assert(test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                 std::basic_string<CharT>,
                                 std::ios_base::openmode,
                                 test_allocator<CharT>>());

  // Non-convertible to std::`basic_string_view<CharT>`
  static_assert(!test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  int,
                                  std::ios_base::openmode,
                                  test_allocator<CharT>>());
  static_assert(!test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  const int&,
                                  std::ios_base::openmode,
                                  test_allocator<CharT>>());
  static_assert(!test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  SomeObject,
                                  std::ios_base::openmode,
                                  test_allocator<CharT>>());
  static_assert(!test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  const SomeObject&,
                                  std::ios_base::openmode,
                                  test_allocator<CharT>>());

  // Mode
  static_assert(!test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  std::basic_string<CharT>,
                                  SomeObject,
                                  NonAllocator>());

  // Allocator
  static_assert(!test_convertible<std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  std::basic_string<CharT>,
                                  std::ios_base::openmode,
                                  NonAllocator>());
}

// #include <concepts>
// #include <type_traits>

// static_assert(std::is_constructible_v<std::istringstream, std::string_view>);
// static_assert(test_convertible<std::basic_istringstream<char, std::char_traits<char>>,
//                                const char*,
//                                std::ios_base::openmode>());
// static_assert(std::is_constructible_v<std::istringstream, int, test_allocator<int>>);

// static_assert(HasMember<std::istringstream, std::string_view, test_allocator<char>>);

//  std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>
// static_assert(HasMember<std::basic_istringstream<char, std::char_traits<char>, test_allocator<char>>,
//                         const char*,
//                         test_allocator<char>>);

// // std::istringstream
// static_assert(HasMember<std::istringstream, std::string_view>);
// #  ifndef TEST_HAS_NO_WIDE_CHARACTERS
// static_assert(!HasMember<std::istringstream, std::wstring_view>);
// #  endif
// static_assert(HasMember<std::istringstream, const char*>);
// #  ifndef TEST_HAS_NO_WIDE_CHARACTERS
// static_assert(!HasMember<std::istringstream, const wchar_t*>);
// #  endif
// static_assert(HasMember<std::istringstream, std::string>);
// #  ifndef TEST_HAS_NO_WIDE_CHARACTERS
// static_assert(!HasMember<std::istringstream, std::wstring>);
// #  endif
// static_assert(!HasMember<std::istringstream, std::array<char, 1>>);
// static_assert(!HasMember<std::istringstream, std::array<char, 0>>);
// // static_assert(!HasMember<std::istringstream, char>);
// // static_assert(!HasMember<std::istringstream, int>);
// static_assert(!HasMember<std::istringstream, SomeObject>);
// static_assert(!HasMember<std::istringstream, std::nullptr_t>);

// // std::wistringstream

// #  ifndef TEST_HAS_NO_WIDE_CHARACTERS
// static_assert(HasMember<std::wistringstream, std::wstring_view>);
// static_assert(!HasMember<std::wistringstream, std::string_view>);
// static_assert(HasMember<std::wistringstream, const wchar_t*>);
// static_assert(!HasMember<std::wistringstream, const char*>);
// static_assert(HasMember<std::wistringstream, std::wstring>);
// static_assert(!HasMember<std::wistringstream, std::string>);
// static_assert(!HasMember<std::istringstream, std::array<wchar_t, 0>>);
// // static_assert(!HasMember<std::wistringstream, wchar_t>);
// // static_assert(!HasMember<std::wistringstream, int>);
// static_assert(!HasMember<std::wistringstream, SomeObject>);
// static_assert(!HasMember<std::wistringstream, std::nullptr_t>);
// #  endif

#define CS(S) MAKE_CSTRING(CharT, S)
#define ST(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
static void test() {
  const test_allocator<CharT> ca;

  // const CharT*
  {
    const std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(
        CS("zmt"), std::ios_base::binary, ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
  // std::basic_string_view<CharT>
  {
    const auto csv = SV("zmt");
    const std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(
        csv, std::ios_base::binary, ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
  // std::basic_string<CharT>
  {
    const auto cs = ST("zmt");
    const std::basic_istringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(
        cs, std::ios_base::binary, ca);
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
