//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: no-localization
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// REQUIRES: locale.fr_FR.UTF-8

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// <thread>

// class thread::id

// template<class charT, class traits>
// basic_ostream<charT, traits>&
// operator<<(basic_ostream<charT, traits>& out, thread::id id);

#include <cassert>
#include <format>
#include <locale>
#include <sstream>
#include <thread>

#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"

template <class CharT>
static void basic() {
  std::thread::id id0 = std::this_thread::get_id();
  std::basic_ostringstream<CharT> os;
  os << id0;

#if TEST_STD_VER > 20
  // C++23 added a formatter specialization for thread::id.
  // This changed the requirement of ostream to have a
  // [thread.thread.id]/2
  //   The text representation for the character type charT of an object of
  //   type thread::id is an unspecified sequence of charT ...
  // This definition is used for both streaming and formatting.
  //
  // Test whether the output is identical.
  std::basic_string<CharT> s = std::format(MAKE_STRING_VIEW(CharT, "{}"), id0);
  assert(s == os.str());
#endif
}

template <class CharT>
static std::basic_string<CharT> format(std::ios_base::fmtflags flags) {
  std::basic_stringstream<CharT> sstr;
  sstr.flags(flags);
  sstr << std::this_thread::get_id();
  return sstr.str();
}

template <class CharT>
static void stream_state() {
  std::basic_stringstream<CharT> sstr;
  sstr << std::this_thread::get_id();
  std::basic_string<CharT> expected = sstr.str();

  // Unaffected by fill, width, and align.

  assert(expected == format<CharT>(std::ios_base::dec | std::ios_base::skipws)); // default flags

  assert(expected == format<CharT>(std::ios_base::oct));
  assert(expected == format<CharT>(std::ios_base::hex));

  assert(expected == format<CharT>(std::ios_base::scientific));
  assert(expected == format<CharT>(std::ios_base::fixed));

  assert(expected == format<CharT>(std::ios_base::boolalpha));
  assert(expected == format<CharT>(std::ios_base::showbase));
  assert(expected == format<CharT>(std::ios_base::showpoint));
  assert(expected == format<CharT>(std::ios_base::showpos));
  assert(expected == format<CharT>(std::ios_base::skipws));  // added for completeness
  assert(expected == format<CharT>(std::ios_base::unitbuf)); // added for completeness
  assert(expected == format<CharT>(std::ios_base::uppercase));

  // Test fill, width, and align.

  sstr.str(std::basic_string<CharT>());
  sstr.fill(CharT('#'));
  sstr.width(expected.size() + 10); // Make sure fill and align affect the output.
  sstr.flags(std::ios_base::dec | std::ios_base::skipws | std::ios_base::right);
  sstr << std::this_thread::get_id();
  expected = sstr.str();

  sstr.str(std::basic_string<CharT>());
  sstr.fill(CharT('*'));
  sstr.width(expected.size());
  sstr.flags(std::ios_base::dec | std::ios_base::skipws | std::ios_base::right);
  sstr << std::this_thread::get_id();
  assert(expected != sstr.str());

  sstr.str(std::basic_string<CharT>());
  sstr.fill(CharT('#'));
  sstr.width(expected.size() - 1);
  sstr.flags(std::ios_base::dec | std::ios_base::skipws | std::ios_base::right);
  sstr << std::this_thread::get_id();
  assert(expected != sstr.str());

  sstr.str(std::basic_string<CharT>());
  sstr.fill(CharT('#'));
  sstr.width(expected.size());
  sstr.flags(std::ios_base::dec | std::ios_base::skipws | std::ios_base::left);
  sstr << std::this_thread::get_id();
  assert(expected != sstr.str());

  sstr.str(std::basic_string<CharT>());
  sstr.fill(CharT('#'));
  sstr.width(expected.size());
  sstr.flags(std::ios_base::dec | std::ios_base::skipws | std::ios_base::internal);
  sstr << std::this_thread::get_id();
  assert(expected == sstr.str()); // internal does *not* affect strings

  // Test the locale's numpunct.

  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));
  sstr.str(std::basic_string<CharT>());
  sstr.fill(CharT('#'));
  sstr.width(expected.size());
  sstr << std::this_thread::get_id();
  assert(expected == sstr.str());
}

template <class CharT>
static void test() {
  basic<CharT>();
  stream_state<CharT>();
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
