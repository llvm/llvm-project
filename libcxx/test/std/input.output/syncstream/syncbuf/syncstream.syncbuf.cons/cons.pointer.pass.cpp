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

// explicit basic_syncbuf(streambuf_type* obuf);

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

    static_assert(!std::convertible_to<std::basic_streambuf<CharT>*, Buf>);
    static_assert(std::constructible_from<Buf, std::basic_streambuf<CharT>*>);

    {
      Buf buf{nullptr};
      assert(buf.get_wrapped() == nullptr);
      assert(buf.get_allocator() == std::allocator<CharT>());
    }
    {
      Buf w;
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
      {
        Buf buf{&w};
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
        assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 1);
#endif
        assert(buf.get_wrapped() == &w);
        assert(buf.get_allocator() == std::allocator<CharT>());
      }
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
    }
  }

  {
    using Buf = std::basic_syncbuf<CharT, constexpr_char_traits<CharT>>;

    static_assert(!std::convertible_to<std::basic_streambuf<CharT, constexpr_char_traits<CharT>>*, Buf>);
    static_assert(std::constructible_from<Buf, std::basic_streambuf<CharT, constexpr_char_traits<CharT>>*>);

    {
      Buf buf{nullptr};
      assert(buf.get_wrapped() == nullptr);
      assert(buf.get_allocator() == std::allocator<CharT>());
    }
    {
      Buf w;
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
      {
        Buf buf{&w};
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
        assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 1);
#endif
        assert(buf.get_wrapped() == &w);
        assert(buf.get_allocator() == std::allocator<CharT>());
      }
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
    }
  }

  {
    using Buf = std::basic_syncbuf<CharT, constexpr_char_traits<CharT>, test_allocator<CharT>>;

    static_assert(!std::convertible_to<std::basic_streambuf<CharT, constexpr_char_traits<CharT>>*, Buf>);
    static_assert(std::constructible_from<Buf, std::basic_streambuf<CharT, constexpr_char_traits<CharT>>*>);

    {
      Buf buf{nullptr};
      assert(buf.get_wrapped() == nullptr);
      assert(buf.get_allocator() == test_allocator<CharT>());
    }
    {
      Buf w;
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
      {
        Buf buf{&w};
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
        assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 1);
#endif
        assert(buf.get_wrapped() == &w);
        assert(buf.get_allocator() == test_allocator<CharT>());
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
