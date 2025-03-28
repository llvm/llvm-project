//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_stringbuf

// explicit basic_stringbuf(ios_base::openmode which = ios_base::in | ios_base::out); // before C++20
// basic_stringbuf() : basic_stringbuf(ios_base::in | ios_base::out) {}               // C++20
// explicit basic_stringbuf(ios_base::openmode which);                                // C++20

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <sstream>
#include <cassert>

#include "test_macros.h"
#if TEST_STD_VER >= 11
#include "test_convertible.h"
#endif

template<typename CharT>
struct testbuf
    : std::basic_stringbuf<CharT>
{
    void check()
    {
      // LWG2995
      //   It is implementation-defined whether the sequence pointers (eback(),
      //   gptr(), egptr(), pbase(), pptr(), epptr()) are initialized to null
      //   pointers.
      // This tests the libc++ specific implementation.
      LIBCPP_ASSERT(this->eback() != nullptr);
      LIBCPP_ASSERT(this->gptr() != nullptr);
      LIBCPP_ASSERT(this->egptr() != nullptr);
      LIBCPP_ASSERT(this->pbase() != nullptr);
      LIBCPP_ASSERT(this->pptr() != nullptr);
      LIBCPP_ASSERT(this->epptr() != nullptr);
      assert(this->str().empty());
    }
};

int main(int, char**)
{
    {
        std::stringbuf buf;
        assert(buf.str() == "");
    }
    {
        testbuf<char> buf;
        buf.check();
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wstringbuf buf;
        assert(buf.str() == L"");
    }
    {
        testbuf<wchar_t> buf;
        buf.check();
    }
#endif

#if TEST_STD_VER >= 11
    {
      typedef std::stringbuf B;
      static_assert(test_convertible<B>(), "");
      static_assert(!test_convertible<B, std::ios_base::openmode>(), "");
    }
#endif

    return 0;
}
