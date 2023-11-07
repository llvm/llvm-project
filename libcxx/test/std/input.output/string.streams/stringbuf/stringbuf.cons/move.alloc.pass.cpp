//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT>>
// class basic_stringbuf

// basic_stringbuf(basic_stringbuf&& rhs, const Allocator& a);

#include <sstream>
#include <cassert>
#include <utility>

#include "make_string.h"
#include "test_allocator.h"
#include "test_macros.h"

#define STR(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
struct test_stringbuf : std::basic_stringbuf<CharT, std::char_traits<CharT>, test_allocator<CharT>> {
  using std::basic_stringbuf<CharT, std::char_traits<CharT>, test_allocator<CharT>>::basic_stringbuf;

  // Checks the following requirement after being moved from:
  //    The six pointers of std::basic_streambuf in *this are guaranteed to be different
  //    from the corresponding pointers in the moved-from rhs unless null.
  void check_different_pointers(test_stringbuf<CharT> const& other) const {
    assert(this->eback() == nullptr || this->eback() != other.eback());
    assert(this->gptr() == nullptr || this->gptr() != other.gptr());
    assert(this->egptr() == nullptr || this->egptr() != other.egptr());
    assert(this->pbase() == nullptr || this->pbase() != other.pbase());
    assert(this->pptr() == nullptr || this->pptr() != other.pptr());
    assert(this->epptr() == nullptr || this->epptr() != other.epptr());
  }
};

template <class CharT>
void test() {
  std::basic_string<CharT> strings[] = {STR(""), STR("short"), STR("loooooooooooooooooooong")};
  for (std::basic_string<CharT> const& s : strings) {
    using StringBuf = std::basic_stringbuf<CharT, std::char_traits<CharT>, test_allocator<CharT>>;
    {
      test_stringbuf<CharT> buf1(s);
      const test_allocator<CharT> a(2);
      test_stringbuf<CharT> buf(std::move(static_cast<StringBuf&>(buf1)), a);
      assert(buf.get_allocator() == a);
      assert(buf.view() == s);
      assert(buf1.view().empty());
      buf.check_different_pointers(buf1);
    }
    {
      test_stringbuf<CharT> buf1(s, std::ios_base::in);
      const test_allocator<CharT> a(2);
      test_stringbuf<CharT> buf(std::move(static_cast<StringBuf&>(buf1)), a);
      assert(buf.get_allocator() == a);
      assert(buf.view() == s);
      assert(buf1.view().empty());
      buf.check_different_pointers(buf1);
    }
    {
      test_stringbuf<CharT> buf1(s, std::ios_base::out);
      const test_allocator<CharT> a(2);
      test_stringbuf<CharT> buf(std::move(static_cast<StringBuf&>(buf1)), a);
      assert(buf.get_allocator() == a);
      assert(buf.view() == s);
      assert(buf1.view().empty());
      buf.check_different_pointers(buf1);
    }
    {
      test_stringbuf<CharT> buf1;
      const test_allocator<CharT> a(2);
      test_stringbuf<CharT> buf(std::move(static_cast<StringBuf&>(buf1)), a);
      assert(buf.get_allocator() == a);
      assert(buf.view().empty());
      assert(buf1.view().empty());
      buf.check_different_pointers(buf1);
    }
    // Use the constructor from an actual std::stringbuf, not test_stringbuf
    {
      StringBuf buf1(s);
      const test_allocator<CharT> a(2);
      StringBuf buf(std::move(buf1), a);
      assert(buf.get_allocator() == a);
      assert(buf.view() == s);
      assert(buf1.view().empty());
    }
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  return 0;
}
