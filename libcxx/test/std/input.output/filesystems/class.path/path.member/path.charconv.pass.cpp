//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: availability-filesystem-missing
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <filesystem>

// class path

// Test constructors, accessors and modifiers that convert from/to various
// character encodings. Constructors and modifiers (append, concat,
// operator/=, operator+=) accept inputs with various character encodings,
// and accessors (*string(), string<>(), u8string()) export the string with
// various encodings.
//
// Some encodings are standardized; char16_t, char32_t and the u8string
// accessor and u8path constructor (and normal functions taking char8_t in
// C++20) convert from/to UTF-16, UTF-32 and UTF-8. wchar_t can be either
// UTF-16 or UTF-32 depending on the size of the wchar_t type, or can be
// left unimplemented.
//
// Plain char is implicitly UTF-8 on posix systems. On Windows, plain char
// is supposed to be in the same encoding as the platform's native file
// system APIs consumes in the functions that take narrow strings as path
// names.

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

#ifdef _WIN32
#  include <windows.h> // SetFileApisToANSI & friends
#endif
namespace fs = std::filesystem;

// Test conversion with strings that fit within the latin1 charset, that fit
// within one code point in UTF-16, and that can be expressible in certain
// one-byte code pages.
static void test_latin_unicode()
{
  const char16_t u16str[] = { 0xe5, 0xe4, 0xf6, 0x00 };
  const char32_t u32str[] = { 0xe5, 0xe4, 0xf6, 0x00 };
  const char str[] = { char(0xc3), char(0xa5), char(0xc3), char(0xa4), char(0xc3), char(0xb6), 0x00 }; // UTF8, in a regular char string
#if TEST_STD_VER > 17 && defined(__cpp_lib_char8_t)
  const char8_t u8str[] = { 0xc3, 0xa5, 0xc3, 0xa4, 0xc3, 0xb6, 0x00 };
#else
  const char u8str[] = { char(0xc3), char(0xa5), char(0xc3), char(0xa4), char(0xc3), char(0xb6), 0x00 };
#endif
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  const wchar_t wstr[] = { 0xe5, 0xe4, 0xf6, 0x00 };
#endif

  // Test well-defined conversion between UTF-8, UTF-16 and UTF-32
  {
    const fs::path p(u16str);
    assert(p.u8string() == u8str);
    assert(p.u16string() == u16str);
    assert(p.u32string() == u32str);
    assert(p.string<char16_t>() == u16str);
    assert(p.string<char32_t>() == u32str);
  }
  {
    const fs::path p(u32str);
    assert(p.u8string() == u8str);
    assert(p.u16string() == u16str);
    assert(p.u32string() == u32str);
    assert(p.string<char16_t>() == u16str);
    assert(p.string<char32_t>() == u32str);
  }
  {
    const fs::path p = fs::u8path(str);
    assert(p.u8string() == u8str);
    assert(p.u16string() == u16str);
    assert(p.u32string() == u32str);
    assert(p.string<char16_t>() == u16str);
    assert(p.string<char32_t>() == u32str);
  }
#if TEST_STD_VER > 17 && defined(__cpp_lib_char8_t)
  {
    // In C++20, the path constructor can unambiguously handle UTF-8 input,
    // even if the plain char constructor would treat it as something else.
    const fs::path p(u8str);
    assert(p.u8string() == u8str);
    assert(p.u16string() == u16str);
    assert(p.u32string() == u32str);
    assert(p.string<char8_t>() == u8str);
    assert(p.string<char16_t>() == u16str);
    assert(p.string<char32_t>() == u32str);
  }
  // Check reading various inputs with string<char8_t>()
  {
    const fs::path p(u16str);
    assert(p.string<char8_t>() == u8str);
  }
  {
    const fs::path p(u32str);
    assert(p.string<char8_t>() == u8str);
  }
  {
    const fs::path p = fs::u8path(str);
    assert(p.string<char8_t>() == u8str);
  }
#endif
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // Test conversion to/from wchar_t.
  {
    const fs::path p(u16str);
    assert(p.wstring() == wstr);
    assert(p.string<wchar_t>() == wstr);
  }
  {
    const fs::path p = fs::u8path(str);
    assert(p.wstring() == wstr);
    assert(p.string<wchar_t>() == wstr);
  }
  {
    const fs::path p(wstr);
    assert(p.wstring() == wstr);
    assert(p.u8string() == u8str);
    assert(p.u16string() == u16str);
    assert(p.u32string() == u32str);
    assert(p.string<wchar_t>() == wstr);
  }
#endif // TEST_HAS_NO_WIDE_CHARACTERS
#ifndef _WIN32
  // Test conversion to/from regular char-based string. On POSIX, this
  // is implied to convert to/from UTF-8.
  {
    const fs::path p(str);
    assert(p.string() == str);
    assert(p.u16string() == u16str);
    assert(p.string<char>() == str);
  }
  {
    const fs::path p(u16str);
    assert(p.string() == str);
    assert(p.string<char>() == str);
  }
#else
  // On windows, the narrow char-based input/output is supposed to be
  // in the charset that narrow file IO APIs use. This can either be the
  // current active code page (ACP) or the OEM code page, exposed by
  // the AreFileApisANSI() function, and settable with SetFileApisToANSI() and
  // SetFileApisToOEM(). We can't set which codepage is active within
  // the process, but for some specific known ones, we can check if they
  // behave as expected.
  SetFileApisToANSI();
  if (GetACP() == 1252) {
    const char latin1[] = { char(0xe5), char(0xe4), char(0xf6), 0x00 };
    {
      const fs::path p(wstr);
      assert(p.string() == latin1);
      assert(p.string<char>() == latin1);
    }
    {
      const fs::path p(latin1);
      assert(p.string() == latin1);
      assert(p.wstring() == wstr);
      assert(p.u8string() == u8str);
      assert(p.u16string() == u16str);
      assert(p.string<char>() == latin1);
      assert(p.string<wchar_t>() == wstr);
    }
  }
  SetFileApisToOEM();
  if (GetOEMCP() == 850 || GetOEMCP() == 437) {
    // These chars are identical in both CP 850 and 437
    const char cp850[] = { char(0x86), char(0x84), char(0x94), 0x00 };
    {
      const fs::path p(wstr);
      assert(p.string() == cp850);
      assert(p.string<char>() == cp850);
    }
    {
      const fs::path p(cp850);
      assert(p.string() == cp850);
      assert(p.wstring() == wstr);
      assert(p.u8string() == u8str);
      assert(p.u16string() == u16str);
      assert(p.string<char>() == cp850);
      assert(p.string<wchar_t>() == wstr);
    }
  }
#endif
}

// Test conversion with strings that don't fit within one UTF-16 code point.
// Here, wchar_t can be either UTF-16 or UTF-32 depending on the size on the
// particular platform.
static void test_wide_unicode()
{
  const char16_t u16str[] = { 0xd801, 0xdc37, 0x00 };
  const char32_t u32str[] = { 0x10437, 0x00 };
#if TEST_STD_VER > 17 && defined(__cpp_lib_char8_t)
  const char8_t u8str[] = { 0xf0, 0x90, 0x90, 0xb7, 0x00 };
#else
  const char u8str[] = { char(0xf0), char(0x90), char(0x90), char(0xb7), 0x00 };
#endif
  const char str[] = { char(0xf0), char(0x90), char(0x90), char(0xb7), 0x00 };
  {
    const fs::path p = fs::u8path(str);
    assert(p.u8string() == u8str);
    assert(p.u16string() == u16str);
    assert(p.u32string() == u32str);
  }
  {
    const fs::path p(u16str);
    assert(p.u8string() == u8str);
    assert(p.u16string() == u16str);
    assert(p.u32string() == u32str);
  }
  {
    const fs::path p(u32str);
    assert(p.u8string() == u8str);
    assert(p.u16string() == u16str);
    assert(p.u32string() == u32str);
  }
#if !defined(TEST_HAS_NO_WIDE_CHARACTERS) && defined(__SIZEOF_WCHAR_T__)
# if __SIZEOF_WCHAR_T__ == 2
  const wchar_t wstr[] = { 0xd801, 0xdc37, 0x00 };
# else
  const wchar_t wstr[] = { 0x10437, 0x00 };
# endif
  // Test conversion to/from wchar_t.
  {
    const fs::path p = fs::u8path(str);
    assert(p.wstring() == wstr);
  }
  {
    const fs::path p(u16str);
    assert(p.wstring() == wstr);
  }
  {
    const fs::path p(u32str);
    assert(p.wstring() == wstr);
  }
  {
    const fs::path p(wstr);
    assert(p.u8string() == u8str);
    assert(p.u16string() == u16str);
    assert(p.u32string() == u32str);
    assert(p.wstring() == wstr);
  }
#endif // !defined(TEST_HAS_NO_WIDE_CHARACTERS) && defined(__SIZEOF_WCHAR_T__)
}

// Test appending paths in different encodings.
static void test_append()
{
  const char16_t u16str[] = { 0xd801, 0xdc37, 0x00 };
  const char32_t u32str[] = { 0x10437, 0x00 };
  const char32_t u32ref[] = { 0x10437, fs::path::preferred_separator, 0x10437, fs::path::preferred_separator, 0x10437, 0x00 };
  const char str[] = { char(0xf0), char(0x90), char(0x90), char(0xb7), 0x00 };
  {
    fs::path p = fs::u8path(str) / u16str / u32str;
    assert(p.u32string() == u32ref);
    p = fs::u8path(str).append(u16str).append(u32str);
    assert(p.u32string() == u32ref);
    p = fs::u8path(str);
    p /= u16str;
    p /= u32str;
    assert(p.u32string() == u32ref);
  }
#if !defined(TEST_HAS_NO_WIDE_CHARACTERS) && defined(__SIZEOF_WCHAR_T__)
# if __SIZEOF_WCHAR_T__ == 2
  const wchar_t wstr[] = { 0xd801, 0xdc37, 0x00 };
# else
  const wchar_t wstr[] = { 0x10437, 0x00 };
# endif
  // Test conversion from wchar_t.
  {
    fs::path p = fs::path(u16str) / wstr / u32str;
    assert(p.u32string() == u32ref);
    p = fs::path(u16str).append(wstr).append(u32str);
    assert(p.u32string() == u32ref);
    p = fs::path(u16str);
    p /= wstr;
    p /= u32str;
    assert(p.u32string() == u32ref);
  }
#endif // !defined(TEST_HAS_NO_WIDE_CHARACTERS) && defined(__SIZEOF_WCHAR_T__)
}

static void test_concat()
{
  const char16_t u16str[] = { 0xd801, 0xdc37, 0x00 };
  const char32_t u32str[] = { 0x10437, 0x00 };
  const char32_t u32ref[] = { 0x10437, 0x10437, 0x10437, 0x00 };
  const char str[] = { char(0xf0), char(0x90), char(0x90), char(0xb7), 0x00 };
  {
    fs::path p = fs::u8path(str);
    p += u16str;
    p += u32str;
    assert(p.u32string() == u32ref);
    p = fs::u8path(str).concat(u16str).concat(u32str);
    assert(p.u32string() == u32ref);
  }
#if !defined(TEST_HAS_NO_WIDE_CHARACTERS) && defined(__SIZEOF_WCHAR_T__)
# if __SIZEOF_WCHAR_T__ == 2
  const wchar_t wstr[] = { 0xd801, 0xdc37, 0x00 };
# else
  const wchar_t wstr[] = { 0x10437, 0x00 };
# endif
  // Test conversion from wchar_t.
  {
    fs::path p = fs::path(u16str);
    p += wstr;
    p += u32str;
    assert(p.u32string() == u32ref);
    p = fs::path(u16str).concat(wstr).concat(u32str);
    assert(p.u32string() == u32ref);
  }
#endif // !defined(TEST_HAS_NO_WIDE_CHARACTERS) && defined(__SIZEOF_WCHAR_T__)
}

static void test_append_concat_narrow()
{
  const char16_t u16str[] = { 0xe5, 0x00 };
  const char32_t u32ref_append[] = { 0xe5, fs::path::preferred_separator, 0xe5, 0x00 };
  const char32_t u32ref_concat[] = { 0xe5, 0xe5, 0x00 };

#if TEST_STD_VER > 17 && defined(__cpp_lib_char8_t)
  {
    const char8_t u8str[] = { 0xc3, 0xa5, 0x00 };
    // In C++20, appends of a char8_t string is unambiguously treated as
    // UTF-8.
    fs::path p = fs::path(u16str) / u8str;
    assert(p.u32string() == u32ref_append);
    p = fs::path(u16str).append(u8str);
    assert(p.u32string() == u32ref_append);
    p = fs::path(u16str);
    p /= u8str;
    assert(p.u32string() == u32ref_append);
    p = fs::path(u16str).concat(u8str);
    assert(p.u32string() == u32ref_concat);
    p = fs::path(u16str);
    p += u8str;
    assert(p.u32string() == u32ref_concat);
  }
#endif
#ifndef _WIN32
  // Test appending a regular char-based string. On POSIX, this
  // is implied to convert to/from UTF-8.
  {
    const char str[] = { char(0xc3), char(0xa5), 0x00 }; // UTF8, in a regular char string
    fs::path p = fs::path(u16str) / str;
    assert(p.u32string() == u32ref_append);
    p = fs::path(u16str).append(str);
    assert(p.u32string() == u32ref_append);
    p = fs::path(u16str);
    p /= str;
    assert(p.u32string() == u32ref_append);
    p = fs::path(u16str).concat(str);
    assert(p.u32string() == u32ref_concat);
    p = fs::path(u16str);
    p += str;
    assert(p.u32string() == u32ref_concat);
  }
#else
  SetFileApisToANSI();
  if (GetACP() == 1252) {
    const char latin1[] = { char(0xe5), 0x00 };
    fs::path p = fs::path(u16str) / latin1;
    assert(p.u32string() == u32ref_append);
    p = fs::path(u16str).append(latin1);
    assert(p.u32string() == u32ref_append);
    p = fs::path(u16str);
    p /= latin1;
    assert(p.u32string() == u32ref_append);
    p = fs::path(u16str).concat(latin1);
    assert(p.u32string() == u32ref_concat);
    p = fs::path(u16str);
    p += latin1;
    assert(p.u32string() == u32ref_concat);
  }
  SetFileApisToOEM();
  if (GetOEMCP() == 850 || GetOEMCP() == 437) {
    // This chars is identical in both CP 850 and 437
    const char cp850[] = { char(0x86), 0x00 };
    fs::path p = fs::path(u16str) / cp850;
    assert(p.u32string() == u32ref_append);
    p = fs::path(u16str).append(cp850);
    assert(p.u32string() == u32ref_append);
    p = fs::path(u16str);
    p /= cp850;
    assert(p.u32string() == u32ref_append);
    p = fs::path(u16str).concat(cp850);
    assert(p.u32string() == u32ref_concat);
    p = fs::path(u16str);
    p += cp850;
    assert(p.u32string() == u32ref_concat);
  }
#endif
}

int main(int, char**)
{
  test_latin_unicode();
  test_wide_unicode();
  test_append();
  test_concat();
  test_append_concat_narrow();

  return 0;
}
