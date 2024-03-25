//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: no-localization
// UNSUPPORTED: !stdlib=libc++ && c++14

// <iomanip>

// quoted

#include <iomanip>
#include <sstream>
#include <string_view>
#include <cassert>

#include "test_macros.h"

bool is_skipws(const std::istream* is) { return (is->flags() & std::ios_base::skipws) != 0; }

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
bool is_skipws(const std::wistream* is) { return (is->flags() & std::ios_base::skipws) != 0; }
#endif

void round_trip(const char* p) {
  std::stringstream ss;
  bool skippingws = is_skipws(&ss);
  std::string_view sv{p};

  ss << std::quoted(sv);
  std::string s;
  ss >> std::quoted(s);
  assert(s == sv);
  assert(skippingws == is_skipws(&ss));
}

void round_trip_ws(const char* p) {
  std::stringstream ss;
  std::noskipws(ss);
  bool skippingws = is_skipws(&ss);
  std::string_view sv{p};

  ss << std::quoted(sv);
  std::string s;
  ss >> std::quoted(s);
  assert(s == sv);
  assert(skippingws == is_skipws(&ss));
}

void round_trip_d(const char* p, char delim) {
  std::stringstream ss;
  std::string_view sv{p};

  ss << std::quoted(sv, delim);
  std::string s;
  ss >> std::quoted(s, delim);
  assert(s == sv);
}

void round_trip_e(const char* p, char escape) {
  std::stringstream ss;
  std::string_view sv{p};

  ss << std::quoted(sv, '"', escape);
  std::string s;
  ss >> std::quoted(s, '"', escape);
  assert(s == sv);
}

std::string quote(const char* p, char delim = '"', char escape = '\\') {
  std::stringstream ss;
  ss << std::quoted(p, delim, escape);
  std::string s;
  ss >> s; // no quote
  return s;
}

std::string unquote(const char* p, char delim = '"', char escape = '\\') {
  std::stringstream ss;
  ss << p;
  std::string s;
  ss >> std::quoted(s, delim, escape);
  return s;
}

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
void round_trip(const wchar_t* p) {
  std::wstringstream ss;
  bool skippingws = is_skipws(&ss);
  std::wstring_view sv{p};

  ss << std::quoted(sv);
  std::wstring s;
  ss >> std::quoted(s);
  assert(s == sv);
  assert(skippingws == is_skipws(&ss));
}

void round_trip_ws(const wchar_t* p) {
  std::wstringstream ss;
  std::noskipws(ss);
  bool skippingws = is_skipws(&ss);
  std::wstring_view sv{p};

  ss << std::quoted(sv);
  std::wstring s;
  ss >> std::quoted(s);
  assert(s == sv);
  assert(skippingws == is_skipws(&ss));
}

void round_trip_d(const wchar_t* p, wchar_t delim) {
  std::wstringstream ss;
  std::wstring_view sv{p};

  ss << std::quoted(sv, delim);
  std::wstring s;
  ss >> std::quoted(s, delim);
  assert(s == sv);
}

void round_trip_e(const wchar_t* p, wchar_t escape) {
  std::wstringstream ss;
  std::wstring_view sv{p};

  ss << std::quoted(sv, wchar_t('"'), escape);
  std::wstring s;
  ss >> std::quoted(s, wchar_t('"'), escape);
  assert(s == sv);
}

std::wstring quote(const wchar_t* p, wchar_t delim = '"', wchar_t escape = '\\') {
  std::wstringstream ss;
  std::wstring_view sv{p};

  ss << std::quoted(sv, delim, escape);
  std::wstring s;
  ss >> s; // no quote
  return s;
}

std::wstring unquote(const wchar_t* p, wchar_t delim = '"', wchar_t escape = '\\') {
  std::wstringstream ss;
  std::wstring_view sv{p};

  ss << sv;
  std::wstring s;
  ss >> std::quoted(s, delim, escape);
  return s;
}
#endif // TEST_HAS_NO_WIDE_CHARACTERS

int main(int, char**) {
  round_trip("");
  round_trip_ws("");
  round_trip_d("", 'q');
  round_trip_e("", 'q');

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  round_trip(L"");
  round_trip_ws(L"");
  round_trip_d(L"", 'q');
  round_trip_e(L"", 'q');
#endif

  round_trip("Hi");
  round_trip_ws("Hi");
  round_trip_d("Hi", '!');
  round_trip_e("Hi", '!');
  assert(quote("Hi", '!') == "!Hi!");
  assert(quote("Hi!", '!') == R"(!Hi\!!)");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  round_trip(L"Hi");
  round_trip_ws(L"Hi");
  round_trip_d(L"Hi", '!');
  round_trip_e(L"Hi", '!');
  assert(quote(L"Hi", '!') == L"!Hi!");
  assert(quote(L"Hi!", '!') == LR"(!Hi\!!)");
#endif

  round_trip("Hi Mom");
  round_trip_ws("Hi Mom");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  round_trip(L"Hi Mom");
  round_trip_ws(L"Hi Mom");
#endif

  assert(quote("") == "\"\"");
  assert(quote("a") == "\"a\"");
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  assert(quote(L"") == L"\"\"");
  assert(quote(L"a") == L"\"a\"");
#endif

  // missing end quote - must not hang
  assert(unquote("\"abc") == "abc");
  assert(unquote("abc") == "abc");     // no delimiter
  assert(unquote("abc def") == "abc"); // no delimiter
  assert(unquote("") == "");           // nothing there

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  assert(unquote(L"\"abc") == L"abc");
  assert(unquote(L"abc") == L"abc");     // no delimiter
  assert(unquote(L"abc def") == L"abc"); // no delimiter
  assert(unquote(L"") == L"");           // nothing there
#endif

  return 0;
}
