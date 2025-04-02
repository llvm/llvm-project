//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// How the constructors of basic_stringbuf initialize the buffer pointers is
// not specified. For some constructors it's implementation defined whether the
// pointers are set to nullptr. Libc++'s implementation directly uses the SSO
// buffer of a std::string as the initial size. This test validates that
// behaviour.
//
// This behaviour is allowed by LWG2995.

#include <sstream>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class CharT>
struct test_buf : public std::basic_stringbuf<CharT> {
  typedef std::basic_streambuf<CharT> base;
  typedef typename base::char_type char_type;
  typedef typename base::int_type int_type;
  typedef typename base::traits_type traits_type;

  char_type* pbase() const { return base::pbase(); }
  char_type* pptr() const { return base::pptr(); }
  char_type* epptr() const { return base::epptr(); }
  void gbump(int n) { base::gbump(n); }

  virtual int_type overflow(int_type c = traits_type::eof()) { return base::overflow(c); }

  test_buf() = default;
  explicit test_buf(std::ios_base::openmode which) : std::basic_stringbuf<CharT>(which) {}

  explicit test_buf(const std::basic_string<CharT>& s) : std::basic_stringbuf<CharT>(s) {}
#if TEST_STD_VER >= 20
  explicit test_buf(const std::allocator<CharT>& a) : std::basic_stringbuf<CharT>(a) {}
  test_buf(std::ios_base::openmode which, const std::allocator<CharT>& a) : std::basic_stringbuf<CharT>(which, a) {}
  explicit test_buf(std::basic_string<CharT>&& s)
      : std::basic_stringbuf<CharT>(std::forward<std::basic_string<CharT>>(s)) {}

  test_buf(const std::basic_string<CharT, std::char_traits<CharT>, min_allocator<CharT>>& s,
           const std::allocator<CharT>& a)
      : std::basic_stringbuf<CharT>(s, a) {}
  test_buf(const std::basic_string<CharT, std::char_traits<CharT>, min_allocator<CharT>>& s,
           std::ios_base::openmode which,
           const std::allocator<CharT>& a)
      : std::basic_stringbuf<CharT>(s, which, a) {}
  test_buf(const std::basic_string<CharT, std::char_traits<CharT>, min_allocator<CharT>>& s)
      : std::basic_stringbuf<CharT>(s) {}
#endif //  TEST_STD_VER >= 20

#if TEST_STD_VER >= 26
  test_buf(std::basic_string_view<CharT> s) : std::basic_stringbuf<CharT>(s) {}
  test_buf(std::basic_string_view<CharT> s, const std::allocator<CharT>& a) : std::basic_stringbuf<CharT>(s, a) {}
  test_buf(std::basic_string_view<CharT> s, std::ios_base::openmode which, const std::allocator<CharT>& a)
      : std::basic_stringbuf<CharT>(s, which, a) {}
#endif //  TEST_STD_VER >= 26
};

template <class CharT>
static void test() {
  std::size_t size = std::basic_string<CharT>().capacity(); // SSO buffer size.
  {
    test_buf<CharT> b;
    assert(b.pbase() != nullptr);
    assert(b.pptr() == b.pbase());
    assert(b.epptr() == b.pbase() + size);
  }
  {
    test_buf<CharT> b(std::ios_base::out);
    assert(b.pbase() != nullptr);
    assert(b.pptr() == b.pbase());
    assert(b.epptr() == b.pbase() + size);
  }
  {
    std::basic_string<CharT> s;
    s.reserve(1024);
    test_buf<CharT> b(s);
    assert(b.pbase() != nullptr);
    assert(b.pptr() == b.pbase());
    assert(b.epptr() == b.pbase() + size); // copy so uses size
  }
#if TEST_STD_VER >= 20
  {
    test_buf<CharT> b = test_buf<CharT>(std::allocator<CharT>());
    assert(b.pbase() != nullptr);
    assert(b.pptr() == b.pbase());
    assert(b.epptr() == b.pbase() + size);
  }
  {
    test_buf<CharT> b = test_buf<CharT>(std::ios_base::out, std::allocator<CharT>());
    assert(b.pbase() != nullptr);
    assert(b.pptr() == b.pbase());
    assert(b.epptr() == b.pbase() + size);
  }
  {
    std::basic_string<CharT> s;
    s.reserve(1024);
    std::size_t capacity = s.capacity();
    test_buf<CharT> b    = test_buf<CharT>(std::move(s));
    assert(b.pbase() != nullptr);
    assert(b.pptr() == b.pbase());
    assert(b.epptr() >= b.pbase() + capacity); // move so uses s.capacity()
  }
  {
    std::basic_string<CharT, std::char_traits<CharT>, min_allocator<CharT>> s;
    s.reserve(1024);
    test_buf<CharT> b = test_buf<CharT>(s, std::allocator<CharT>());
    assert(b.pbase() != nullptr);
    assert(b.pptr() == b.pbase());
    assert(b.epptr() == b.pbase() + size); // copy so uses size
  }
  {
    std::basic_string<CharT, std::char_traits<CharT>, min_allocator<CharT>> s;
    s.reserve(1024);
    test_buf<CharT> b = test_buf<CharT>(s, std::ios_base::out, std::allocator<CharT>());
    assert(b.pbase() != nullptr);
    assert(b.pptr() == b.pbase());
    assert(b.epptr() == b.pbase() + size); // copy so uses size
  }
  {
    std::basic_string<CharT, std::char_traits<CharT>, min_allocator<CharT>> s;
    s.reserve(1024);
    test_buf<CharT> b = test_buf<CharT>(s);
    assert(b.pbase() != nullptr);
    assert(b.pptr() == b.pbase());
    assert(b.epptr() == b.pbase() + size); // copy so uses size
  }
#endif // TEST_STD_VER >= 20
#if TEST_STD_VER >= 26
  {
    std::basic_string_view<CharT> s;
    test_buf<CharT> b = test_buf<CharT>(s);
    assert(b.pbase() != nullptr);
    assert(b.pptr() == b.pbase());
    assert(b.epptr() == b.pbase() + size);
  }
  {
    std::basic_string_view<CharT> s;
    test_buf<CharT> b = test_buf<CharT>(s, std::allocator<CharT>());
    assert(b.pbase() != nullptr);
    assert(b.pptr() == b.pbase());
    assert(b.epptr() == b.pbase() + size);
  }
  {
    std::basic_string_view<CharT> s;
    test_buf<CharT> b = test_buf<CharT>(s, std::ios_base::out, std::allocator<CharT>());
    assert(b.pbase() != nullptr);
    assert(b.pptr() == b.pbase());
    assert(b.epptr() == b.pbase() + size);
  }
#endif // TEST_STD_VER >= 26
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  return 0;
}
