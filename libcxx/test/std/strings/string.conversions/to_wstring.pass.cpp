//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: no-wide-characters

// <string>

// wstring to_wstring(int val);
// wstring to_wstring(unsigned val);
// wstring to_wstring(long val);
// wstring to_wstring(unsigned long val);
// wstring to_wstring(long long val);
// wstring to_wstring(unsigned long long val);
// wstring to_wstring(float val);
// wstring to_wstring(double val);
// wstring to_wstring(long double val);

#include <cassert>
#include <format>
#include <string>
#include <limits>

#include "parse_integer.h"
#include "test_macros.h"

template <class T>
void test_signed() {
  {
    std::wstring s = std::to_wstring(T(0));
    assert(s.size() == 1);
    assert(s[s.size()] == 0);
    assert(s == L"0");
#if TEST_STD_VER >= 26
    assert(s == std::format(L"{}", T(0)));
#endif
  }
  {
    std::wstring s = std::to_wstring(T(12345));
    assert(s.size() == 5);
    assert(s[s.size()] == 0);
    assert(s == L"12345");
#if TEST_STD_VER >= 26
    assert(s == std::format(L"{}", T(12345)));
#endif
  }
  {
    std::wstring s = std::to_wstring(T(-12345));
    assert(s.size() == 6);
    assert(s[s.size()] == 0);
    assert(s == L"-12345");
#if TEST_STD_VER >= 26
    assert(s == std::format(L"{}", T(-12345)));
#endif
  }
  {
    std::wstring s = std::to_wstring(std::numeric_limits<T>::max());
    assert(s.size() == std::numeric_limits<T>::digits10 + 1);
    T t = parse_integer<T>(s);
    assert(t == std::numeric_limits<T>::max());
#if TEST_STD_VER >= 26
    assert(s == std::format(L"{}", T(std::numeric_limits<T>::max())));
#endif
  }
  {
    std::wstring s = std::to_wstring(std::numeric_limits<T>::min());
    T t            = parse_integer<T>(s);
    assert(t == std::numeric_limits<T>::min());
#if TEST_STD_VER >= 26
    assert(s == std::format(L"{}", T(std::numeric_limits<T>::min())));
#endif
  }
}

template <class T>
void test_unsigned() {
  {
    std::wstring s = std::to_wstring(T(0));
    assert(s.size() == 1);
    assert(s[s.size()] == 0);
    assert(s == L"0");
#if TEST_STD_VER >= 26
    assert(s == std::format(L"{}", T(0)));
#endif
  }
  {
    std::wstring s = std::to_wstring(T(12345));
    assert(s.size() == 5);
    assert(s[s.size()] == 0);
    assert(s == L"12345");
#if TEST_STD_VER >= 26
    assert(s == std::format(L"{}", T(12345)));
#endif
  }
  {
    std::wstring s = std::to_wstring(std::numeric_limits<T>::max());
    assert(s.size() == std::numeric_limits<T>::digits10 + 1);
    T t = parse_integer<T>(s);
    assert(t == std::numeric_limits<T>::max());
#if TEST_STD_VER >= 26
    assert(s == std::format(L"{}", T(std::numeric_limits<T>::max())));
#endif
  }
}

template <class T>
void test_float() {
  {
    std::wstring s = std::to_wstring(T(0));
#if TEST_STD_VER < 26
    assert(s.size() == 8);
    assert(s[s.size()] == 0);
    assert(s == L"0.000000");
#else
    std::wstring f = std::format(L"{}", T(0));
    assert(s == f);
    assert(s == L"0");
#endif
  }
  {
    std::wstring s = std::to_wstring(T(12345));
#if TEST_STD_VER < 26
    assert(s.size() == 12);
    assert(s[s.size()] == 0);
    assert(s == L"12345.000000");
#else
    std::wstring f = std::format(L"{}", T(12345));
    assert(s == f);
    assert(s == L"12345");
#endif
  }
  {
    std::wstring s = std::to_wstring(T(-12345));
#if TEST_STD_VER < 26
    assert(s.size() == 13);
    assert(s[s.size()] == 0);
    assert(s == L"-12345.000000");
#else
    std::wstring f = std::format(L"{}", T(-12345));
    assert(s == f);
    assert(s == L"-12345");
#endif
  }

#if TEST_STD_VER >= 26
  {
    std::wstring s = std::to_wstring(T(90.84));
    std::wstring f = std::format(L"{}", T(90.84));
    assert(s == f);
    assert(s == L"90.84");
  }
  {
    std::wstring s = std::to_wstring(T(-90.84));
    std::wstring f = std::format(L"{}", T(-90.84));
    assert(s == f);
    assert(s == L"-90.84");
  }
#endif
}

#if TEST_STD_VER >= 26

template <class T>
void test_float_with_locale(const char* locale, T inputValue, const wchar_t* expectedValue) {
  setlocale(LC_ALL, locale);

  std::wstring s = std::to_wstring(inputValue);
  std::wstring f = std::format(L"{}", inputValue);
  assert(s == f);
  assert(s == expectedValue);
}

void test_float_with_locale() {
  // Locale "C"

  test_float_with_locale<float>("C", 0.9084, L"0.9084");
  test_float_with_locale<double>("C", 0.9084, L"0.9084");
  test_float_with_locale<long double>("C", 0.9084, L"0.9084");

  test_float_with_locale<float>("C", -0.9084, L"-0.9084");
  test_float_with_locale<double>("C", -0.9084, L"-0.9084");
  test_float_with_locale<long double>("C", -0.9084, L"-0.9084");

  test_float_with_locale<float>("C", 1e-7, L"1e-07");
  test_float_with_locale<double>("C", 1e-7, L"1e-07");
  test_float_with_locale<long double>("C", 1e-7, L"1e-07");

  test_float_with_locale<float>("C", -1e-7, L"-1e-07");
  test_float_with_locale<double>("C", -1e-7, L"-1e-07");
  test_float_with_locale<long double>("C", -1e-7, L"-1e-07");

  test_float_with_locale<float>("C", 1.7976931348623157e+308, L"inf");
  test_float_with_locale<double>("C", 1.7976931348623157e+308, L"1.7976931348623157e+308");
  test_float_with_locale<long double>("C", 1.7976931348623157e+308, L"1.7976931348623157e+308");

  test_float_with_locale<float>("C", -1.7976931348623157e+308, L"-inf");
  test_float_with_locale<double>("C", -1.7976931348623157e+308, L"-1.7976931348623157e+308");
  test_float_with_locale<long double>("C", -1.7976931348623157e+308, L"-1.7976931348623157e+308");

  // Locale "uk_UA.UTF-8"

  test_float_with_locale<float>("uk_UA.UTF-8", 0.9084, L"0.9084");
  test_float_with_locale<double>("uk_UA.UTF-8", 0.9084, L"0.9084");
  test_float_with_locale<double>("uk_UA.UTF-8", 0.9084, L"0.9084");

  test_float_with_locale<float>("uk_UA.UTF-8", -0.9084, L"-0.9084");
  test_float_with_locale<double>("uk_UA.UTF-8", -0.9084, L"-0.9084");
  test_float_with_locale<long double>("uk_UA.UTF-8", -0.9084, L"-0.9084");

  test_float_with_locale<float>("uk_UA.UTF-8", 1e-7, L"1e-07");
  test_float_with_locale<double>("uk_UA.UTF-8", 1e-7, L"1e-07");
  test_float_with_locale<long double>("uk_UA.UTF-8", 1e-7, L"1e-07");

  test_float_with_locale<float>("uk_UA.UTF-8", -1e-7, L"-1e-07");
  test_float_with_locale<double>("uk_UA.UTF-8", -1e-7, L"-1e-07");
  test_float_with_locale<long double>("uk_UA.UTF-8", -1e-7, L"-1e-07");

  test_float_with_locale<float>("uk_UA.UTF-8", 1.7976931348623157e+308, L"inf");
  test_float_with_locale<double>("uk_UA.UTF-8", 1.7976931348623157e+308, L"1.7976931348623157e+308");
  test_float_with_locale<long double>("uk_UA.UTF-8", 1.7976931348623157e+308, L"1.7976931348623157e+308");

  test_float_with_locale<float>("uk_UA.UTF-8", -1.7976931348623157e+308, L"-inf");
  test_float_with_locale<double>("uk_UA.UTF-8", -1.7976931348623157e+308, L"-1.7976931348623157e+308");
  test_float_with_locale<long double>("uk_UA.UTF-8", -1.7976931348623157e+308, L"-1.7976931348623157e+308");
}

#endif

int main(int, char**) {
  test_signed<int>();
  test_signed<long>();
  test_signed<long long>();
  test_unsigned<unsigned>();
  test_unsigned<unsigned long>();
  test_unsigned<unsigned long long>();
  test_float<float>();
  test_float<double>();
  test_float<long double>();
#if TEST_STD_VER >= 26
  test_float_with_locale();
#endif

  return 0;
}
