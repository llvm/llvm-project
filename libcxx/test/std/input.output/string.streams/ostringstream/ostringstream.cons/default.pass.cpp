//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_ostringstream

// explicit basic_ostringstream(ios_base::openmode which = ios_base::out); // before C++20
// basic_ostringstream() : basic_ostringstream(ios_base::out) {}           // C++20
// explicit basic_ostringstream(ios_base::openmode which);                 // C++20

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <sstream>
#include <cassert>

#include "operator_hijacker.h"
#include "test_macros.h"
#if TEST_STD_VER >= 11
#include "test_convertible.h"

template <typename S>
void test() {
  static_assert(test_convertible<S>(), "");
  static_assert(!test_convertible<S, std::ios_base::openmode>(), "");
}
#endif

int main(int, char**)
{
    {
        std::ostringstream ss;
        assert(ss.rdbuf() != nullptr);
        assert(ss.good());
        assert(ss.str() == "");
    }
    {
      std::basic_ostringstream<char, std::char_traits<char>, operator_hijacker_allocator<char> > ss;
      assert(ss.rdbuf() != nullptr);
      assert(ss.good());
      assert(ss.str() == "");
    }
    {
        std::ostringstream ss(std::ios_base::out);
        assert(ss.rdbuf() != nullptr);
        assert(ss.good());
        assert(ss.str() == "");
    }
    {
      std::basic_ostringstream<char, std::char_traits<char>, operator_hijacker_allocator<char> > ss(std::ios_base::out);
      assert(ss.rdbuf() != nullptr);
      assert(ss.good());
      assert(ss.str() == "");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wostringstream ss;
        assert(ss.rdbuf() != nullptr);
        assert(ss.good());
        assert(ss.str() == L"");
    }
    {
      std::basic_ostringstream<wchar_t, std::char_traits<wchar_t>, operator_hijacker_allocator<wchar_t> > ss;
      assert(ss.rdbuf() != nullptr);
      assert(ss.good());
      assert(ss.str() == L"");
    }
    {
        std::wostringstream ss(std::ios_base::out);
        assert(ss.rdbuf() != nullptr);
        assert(ss.good());
        assert(ss.str() == L"");
    }
    {
      std::basic_ostringstream<wchar_t, std::char_traits<wchar_t>, operator_hijacker_allocator<wchar_t> > ss(
          std::ios_base::out);
      assert(ss.rdbuf() != nullptr);
      assert(ss.good());
      assert(ss.str() == L"");
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

#if TEST_STD_VER >= 11
    test<std::ostringstream>();
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::wostringstream>();
#   endif
#endif

    return 0;
}
