//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_istringstream

// explicit basic_istringstream(ios_base::openmode which = ios_base::in); // before C++20
// basic_istringstream() : basic_istringstream(ios_base::in) {}           // C++20
// explicit basic_istringstream(ios_base::openmode which);                // C++20

#include <sstream>
#include <cassert>

#include "test_macros.h"
#include "operator_hijacker.h"
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
        std::istringstream ss;
        assert(ss.rdbuf() != nullptr);
        assert(ss.good());
        assert(ss.str() == "");
    }
    {
      std::basic_istringstream<char, std::char_traits<char>, operator_hijacker_allocator<char> > ss;
      assert(ss.rdbuf() != nullptr);
      assert(ss.good());
      assert(ss.str() == "");
    }
    {
        std::istringstream ss(std::ios_base::in);
        assert(ss.rdbuf() != nullptr);
        assert(ss.good());
        assert(ss.str() == "");
    }
    {
      std::basic_istringstream<char, std::char_traits<char>, operator_hijacker_allocator<char> > ss(std::ios_base::in);
      assert(ss.rdbuf() != nullptr);
      assert(ss.good());
      assert(ss.str() == "");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wistringstream ss;
        assert(ss.rdbuf() != nullptr);
        assert(ss.good());
        assert(ss.str() == L"");
    }
    {
        std::wistringstream ss(std::ios_base::in);
        assert(ss.rdbuf() != nullptr);
        assert(ss.good());
        assert(ss.str() == L"");
    }
    {
      std::basic_istringstream<wchar_t, std::char_traits<wchar_t>, operator_hijacker_allocator<wchar_t> > ss;
      assert(ss.rdbuf() != nullptr);
      assert(ss.good());
      assert(ss.str() == L"");
    }
    {
      std::basic_istringstream<wchar_t, std::char_traits<wchar_t>, operator_hijacker_allocator<wchar_t> > ss(
          std::ios_base::in);
      assert(ss.rdbuf() != nullptr);
      assert(ss.good());
      assert(ss.str() == L"");
    }
#endif

#if TEST_STD_VER >= 11
    test<std::istringstream>();
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::wistringstream>();
#   endif
#endif

    return 0;
}
