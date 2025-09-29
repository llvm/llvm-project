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

// explicit basic_stringstream(const basic_string<charT,traits,Allocator>& str,
//                             ios_base::openmode which = ios_base::out|ios_base::in);

#include <sstream>
#include <cassert>

#include "test_macros.h"
#include "operator_hijacker.h"

template<typename T>
struct NoDefaultAllocator : std::allocator<T>
{
  template<typename U> struct rebind { using other = NoDefaultAllocator<U>; };
  NoDefaultAllocator(int id_) : id(id_) { }
  template<typename U> NoDefaultAllocator(const NoDefaultAllocator<U>& a) : id(a.id) { }
  int id;
};


int main(int, char**)
{
    {
        std::stringstream ss(" 123 456 ");
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
      std::basic_stringstream<char, std::char_traits<char>, operator_hijacker_allocator<char> > ss(" 123 456 ");
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
        std::wstringstream ss(L" 123 456 ");
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
      std::basic_stringstream<wchar_t, std::char_traits<wchar_t>, operator_hijacker_allocator<wchar_t> > ss(
          L" 123 456 ");
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
    { // This is https://llvm.org/PR33727
        typedef std::basic_string   <char, std::char_traits<char>, NoDefaultAllocator<char> > S;
        typedef std::basic_stringbuf<char, std::char_traits<char>, NoDefaultAllocator<char> > SB;

        S s(NoDefaultAllocator<char>(1));
        SB sb(s);
        // This test is not required by the standard, but *where else* could it get the allocator?
        assert(sb.str().get_allocator() == s.get_allocator());
    }

  return 0;
}
