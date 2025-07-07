//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_stringstream

// basic_stringstream(basic_stringstream&& rhs);

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <sstream>
#include <cassert>

#include "test_macros.h"
#include "operator_hijacker.h"

int main(int, char**)
{
    {
        std::stringstream ss0(" 123 456 ");
        std::stringstream ss(std::move(ss0));
        assert(ss.rdbuf() != nullptr);
        assert(ss.good());
        assert(ss.str() == " 123 456 ");
        int i = 0;
        ss >> i;
        assert(i == 123);
        ss >> i;
        assert(i == 456);
        ss << i << ' ' << 123;
        assert(ss.str() == "456 1236 ");
    }
    {
      std::basic_stringstream<char, std::char_traits<char>, operator_hijacker_allocator<char> > ss0(" 123 456 ");
      std::basic_stringstream<char, std::char_traits<char>, operator_hijacker_allocator<char> > ss(std::move(ss0));
      assert(ss.rdbuf() != nullptr);
      assert(ss.good());
      assert(ss.str() == " 123 456 ");
      int i = 0;
      ss >> i;
      assert(i == 123);
      ss >> i;
      assert(i == 456);
      ss << i << ' ' << 123;
      assert(ss.str() == "456 1236 ");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wstringstream ss0(L" 123 456 ");
        std::wstringstream ss(std::move(ss0));
        assert(ss.rdbuf() != nullptr);
        assert(ss.good());
        assert(ss.str() == L" 123 456 ");
        int i = 0;
        ss >> i;
        assert(i == 123);
        ss >> i;
        assert(i == 456);
        ss << i << ' ' << 123;
        assert(ss.str() == L"456 1236 ");
    }
    {
      std::basic_stringstream<wchar_t, std::char_traits<wchar_t>, operator_hijacker_allocator<wchar_t> > ss0(
          L" 123 456 ");
      std::basic_stringstream<wchar_t, std::char_traits<wchar_t>, operator_hijacker_allocator<wchar_t> > ss(
          std::move(ss0));
      assert(ss.rdbuf() != nullptr);
      assert(ss.good());
      assert(ss.str() == L" 123 456 ");
      int i = 0;
      ss >> i;
      assert(i == 123);
      ss >> i;
      assert(i == 456);
      ss << i << ' ' << 123;
      assert(ss.str() == L"456 1236 ");
    }
#endif

  return 0;
}
