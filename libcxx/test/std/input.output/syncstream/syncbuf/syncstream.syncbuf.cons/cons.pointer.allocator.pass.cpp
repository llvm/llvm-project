//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-experimental-syncstream

// <syncstream>

// template <class charT, class traits, class Allocator>
// class basic_syncbuf;

// basic_syncbuf(streambuf_type* obuf, const Allocator&);

#include <cassert>
#include <concepts>
#include <syncstream>

#include "test_macros.h"
#include "constexpr_char_traits.h"
#include "test_allocator.h"

template <class CharT>
void test() {
  {
    using Buf = std::basic_syncbuf<CharT>;
    const std::allocator<CharT> alloc;
    {
      Buf buf = {nullptr, alloc};
      assert(buf.get_wrapped() == nullptr);
      assert(buf.get_allocator() == alloc);
    }
    {
      Buf w;
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
      {
        Buf buf = {&w, alloc};
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
        assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 1);
#endif
        assert(buf.get_wrapped() == &w);
        assert(buf.get_allocator() == alloc);
      }
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
    }
  }

  {
    using Buf = std::basic_syncbuf<CharT, constexpr_char_traits<CharT>>;
    const std::allocator<CharT> alloc;
    {
      Buf buf = {nullptr, alloc};
      assert(buf.get_wrapped() == nullptr);
      assert(buf.get_allocator() == alloc);
    }
    {
      Buf w;
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
      {
        Buf buf{&w, alloc};
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
        assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 1);
#endif
        assert(buf.get_wrapped() == &w);
        assert(buf.get_allocator() == alloc);
      }
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
    }
  }

  {
    using Buf = std::basic_syncbuf<CharT, constexpr_char_traits<CharT>, test_allocator<CharT>>;
    test_allocator<CharT> alloc{42};
    {
      Buf buf = {nullptr, alloc};
      assert(buf.get_wrapped() == nullptr);
      assert(buf.get_allocator() == alloc);
    }
    {
      Buf w;
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
      {
        Buf buf{&w, alloc};
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
        assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 1);
#endif
        assert(buf.get_wrapped() == &w);
        assert(buf.get_allocator() == alloc);
      }
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
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
