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

#include <string>
#include <cassert>
#include <limits>

#include "parse_integer.h"
#include "test_macros.h"

template <class T>
void
test_signed()
{
    {
        std::wstring s = std::to_wstring(T(0));
        assert(s.size() == 1);
        assert(s[s.size()] == 0);
        assert(s == L"0");
    }
    {
        std::wstring s = std::to_wstring(T(12345));
        assert(s.size() == 5);
        assert(s[s.size()] == 0);
        assert(s == L"12345");
    }
    {
        std::wstring s = std::to_wstring(T(-12345));
        assert(s.size() == 6);
        assert(s[s.size()] == 0);
        assert(s == L"-12345");
    }
    {
        std::wstring s = std::to_wstring(std::numeric_limits<T>::max());
        assert(s.size() == std::numeric_limits<T>::digits10 + 1);
        T t = parse_integer<T>(s);
        assert(t == std::numeric_limits<T>::max());
    }
    {
        std::wstring s = std::to_wstring(std::numeric_limits<T>::min());
        T t = parse_integer<T>(s);
        assert(t == std::numeric_limits<T>::min());
    }
}

template <class T>
void
test_unsigned()
{
    {
        std::wstring s = std::to_wstring(T(0));
        assert(s.size() == 1);
        assert(s[s.size()] == 0);
        assert(s == L"0");
    }
    {
        std::wstring s = std::to_wstring(T(12345));
        assert(s.size() == 5);
        assert(s[s.size()] == 0);
        assert(s == L"12345");
    }
    {
        std::wstring s = std::to_wstring(std::numeric_limits<T>::max());
        assert(s.size() == std::numeric_limits<T>::digits10 + 1);
        T t = parse_integer<T>(s);
        assert(t == std::numeric_limits<T>::max());
    }
}

template <class T>
void
test_float()
{
    {
        std::wstring s = std::to_wstring(T(0));
        assert(s.size() == 8);
        assert(s[s.size()] == 0);
        assert(s == L"0.000000");
    }
    {
        std::wstring s = std::to_wstring(T(12345));
        assert(s.size() == 12);
        assert(s[s.size()] == 0);
        assert(s == L"12345.000000");
    }
    {
        std::wstring s = std::to_wstring(T(-12345));
        assert(s.size() == 13);
        assert(s[s.size()] == 0);
        assert(s == L"-12345.000000");
    }
}

int main(int, char**)
{
    test_signed<int>();
    test_signed<long>();
    test_signed<long long>();
    test_unsigned<unsigned>();
    test_unsigned<unsigned long>();
    test_unsigned<unsigned long long>();
    test_float<float>();
    test_float<double>();
    test_float<long double>();

  return 0;
}
