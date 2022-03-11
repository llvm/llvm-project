//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <iomanip>

// std::quoted
//   Verify that the result type of std::quoted can be streamed to
//   (and from) ostreams with the correct CharTraits, and not those
//   with the wrong CharTraits. To avoid our having to create working
//   ostreams with weird CharTraits, this is a compile-only test.

#include <iomanip>
#include <istream>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>

#include "test_allocator.h"
#include "test_macros.h"

template<class IS, class Q>
decltype(std::declval<IS>() >> std::declval<Q>(), std::true_type())
has_rightshift_impl(int) { return std::true_type(); }

template<class IS, class Q>
std::false_type
has_rightshift_impl(long) { return std::false_type(); }

template<class IS, class Q>
struct HasRightShift : decltype(has_rightshift_impl<IS, Q>(0)) {};

template<class OS, class Q>
decltype(std::declval<OS>() << std::declval<Q>(), std::true_type())
has_leftshift_impl(int) { return std::true_type(); }

template<class OS, class Q>
std::false_type
has_leftshift_impl(long) { return std::false_type(); }

template<class OS, class Q>
struct HasLeftShift : decltype(has_leftshift_impl<OS, Q>(0)) {};

template<class CharT>
struct FakeCharTraits : std::char_traits<CharT> {};

void test_string_literal()
{
  using Q = decltype(std::quoted("hello"));
  static_assert( HasLeftShift<std::ostream&, Q>::value, "");
  static_assert(!HasRightShift<std::istream&, Q>::value, "");
  static_assert( HasLeftShift<std::basic_ostream<char, FakeCharTraits<char>>&, Q>::value, "");
  static_assert(!HasRightShift<std::basic_istream<char, FakeCharTraits<char>>&, Q>::value, "");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  using WQ = decltype(std::quoted(L"hello"));
  static_assert( HasLeftShift<std::wostream&, WQ>::value, "");
  static_assert(!HasRightShift<std::wistream&, WQ>::value, "");
  static_assert( HasLeftShift<std::basic_ostream<wchar_t, FakeCharTraits<wchar_t>>&, WQ>::value, "");
  static_assert(!HasRightShift<std::basic_istream<wchar_t, FakeCharTraits<wchar_t>>&, WQ>::value, "");

  static_assert(!HasLeftShift<std::ostream&, WQ>::value, "");
  static_assert(!HasLeftShift<std::wostream&, Q>::value, "");
#endif // TEST_HAS_NO_WIDE_CHARACTERS
}

void test_std_string()
{
  std::string s = "hello";
  const auto& cs = s;
  using Q = decltype(std::quoted(s));
  using CQ = decltype(std::quoted(cs));
  static_assert( HasLeftShift<std::ostream&, Q>::value, "");
  static_assert( HasRightShift<std::istream&, Q>::value, "");
  static_assert( HasLeftShift<std::ostream&, CQ>::value, "");
  static_assert(!HasRightShift<std::istream&, CQ>::value, "");
  static_assert(!HasLeftShift<std::basic_ostream<char, FakeCharTraits<char>>&, Q>::value, "");
  static_assert(!HasRightShift<std::basic_istream<char, FakeCharTraits<char>>&, Q>::value, "");
  static_assert(!HasLeftShift<std::basic_ostream<char, FakeCharTraits<char>>&, CQ>::value, "");
  static_assert(!HasRightShift<std::basic_istream<char, FakeCharTraits<char>>&, CQ>::value, "");

  std::basic_string<char, FakeCharTraits<char>, test_allocator<char>> st = "hello";
  const auto& cst = st;
  using QT = decltype(std::quoted(st));
  using CQT = decltype(std::quoted(cst));
  static_assert(!HasLeftShift<std::ostream&, QT>::value, "");
  static_assert(!HasRightShift<std::istream&, QT>::value, "");
  static_assert(!HasLeftShift<std::ostream&, CQT>::value, "");
  static_assert(!HasRightShift<std::istream&, CQT>::value, "");
  static_assert( HasLeftShift<std::basic_ostream<char, FakeCharTraits<char>>&, QT>::value, "");
  static_assert( HasRightShift<std::basic_istream<char, FakeCharTraits<char>>&, QT>::value, "");
  static_assert( HasLeftShift<std::basic_ostream<char, FakeCharTraits<char>>&, CQT>::value, "");
  static_assert(!HasRightShift<std::basic_istream<char, FakeCharTraits<char>>&, CQT>::value, "");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  std::wstring ws = L"hello";
  const auto& cws = ws;
  using WQ = decltype(std::quoted(ws));
  using CWQ = decltype(std::quoted(cws));
  static_assert( HasLeftShift<std::wostream&, WQ>::value, "");
  static_assert( HasRightShift<std::wistream&, WQ>::value, "");
  static_assert( HasLeftShift<std::wostream&, CWQ>::value, "");
  static_assert(!HasRightShift<std::wistream&, CWQ>::value, "");
  static_assert(!HasLeftShift<std::basic_ostream<wchar_t, FakeCharTraits<wchar_t>>&, WQ>::value, "");
  static_assert(!HasRightShift<std::basic_istream<wchar_t, FakeCharTraits<wchar_t>>&, WQ>::value, "");
  static_assert(!HasLeftShift<std::basic_ostream<wchar_t, FakeCharTraits<wchar_t>>&, CWQ>::value, "");
  static_assert(!HasRightShift<std::basic_istream<wchar_t, FakeCharTraits<wchar_t>>&, CWQ>::value, "");

  static_assert(!HasLeftShift<std::ostream&, WQ>::value, "");
  static_assert(!HasLeftShift<std::wostream&, Q>::value, "");
#endif // TEST_HAS_NO_WIDE_CHARACTERS
}

void test_std_string_view()
{
  std::string_view s = "hello";
  const auto& cs = s;
  using Q = decltype(std::quoted(s));
  using CQ = decltype(std::quoted(cs));
  static_assert( HasLeftShift<std::ostream&, Q>::value, "");
  static_assert(!HasRightShift<std::istream&, Q>::value, "");
  static_assert( HasLeftShift<std::ostream&, CQ>::value, "");
  static_assert(!HasRightShift<std::istream&, CQ>::value, "");
  static_assert(!HasLeftShift<std::basic_ostream<char, FakeCharTraits<char>>&, Q>::value, "");
  static_assert(!HasRightShift<std::basic_istream<char, FakeCharTraits<char>>&, Q>::value, "");
  static_assert(!HasLeftShift<std::basic_ostream<char, FakeCharTraits<char>>&, CQ>::value, "");
  static_assert(!HasRightShift<std::basic_istream<char, FakeCharTraits<char>>&, CQ>::value, "");

  std::basic_string_view<char, FakeCharTraits<char>> st = "hello";
  const auto& cst = st;
  using QT = decltype(std::quoted(st));
  using CQT = decltype(std::quoted(cst));
  static_assert(!HasLeftShift<std::ostream&, QT>::value, "");
  static_assert(!HasRightShift<std::istream&, QT>::value, "");
  static_assert(!HasLeftShift<std::ostream&, CQT>::value, "");
  static_assert(!HasRightShift<std::istream&, CQT>::value, "");
  static_assert( HasLeftShift<std::basic_ostream<char, FakeCharTraits<char>>&, QT>::value, "");
  static_assert(!HasRightShift<std::basic_istream<char, FakeCharTraits<char>>&, QT>::value, "");
  static_assert( HasLeftShift<std::basic_ostream<char, FakeCharTraits<char>>&, CQT>::value, "");
  static_assert(!HasRightShift<std::basic_istream<char, FakeCharTraits<char>>&, CQT>::value, "");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  std::wstring_view ws = L"hello";
  const auto& cws = ws;
  using WQ = decltype(std::quoted(ws));
  using CWQ = decltype(std::quoted(cws));
  static_assert( HasLeftShift<std::wostream&, WQ>::value, "");
  static_assert(!HasRightShift<std::wistream&, WQ>::value, "");
  static_assert( HasLeftShift<std::wostream&, CWQ>::value, "");
  static_assert(!HasRightShift<std::wistream&, CWQ>::value, "");
  static_assert(!HasLeftShift<std::basic_ostream<wchar_t, FakeCharTraits<wchar_t>>&, WQ>::value, "");
  static_assert(!HasRightShift<std::basic_istream<wchar_t, FakeCharTraits<wchar_t>>&, WQ>::value, "");
  static_assert(!HasLeftShift<std::basic_ostream<wchar_t, FakeCharTraits<wchar_t>>&, CWQ>::value, "");
  static_assert(!HasRightShift<std::basic_istream<wchar_t, FakeCharTraits<wchar_t>>&, CWQ>::value, "");

  static_assert(!HasLeftShift<std::ostream&, WQ>::value, "");
  static_assert(!HasLeftShift<std::wostream&, Q>::value, "");
#endif // TEST_HAS_NO_WIDE_CHARACTERS
}
