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
// class basic_osyncstream;

// basic_osyncstream(basic_ostream& os, const Allocator& allocator);

#include <cassert>
#include <concepts>
#include <syncstream>
#include <sstream>

#include "test_macros.h"
#include "constexpr_char_traits.h"
#include "test_allocator.h"

template <class CharT>
void test() {
  {
    using OS = std::basic_osyncstream<CharT>;
    using W  = std::basic_ostringstream<CharT>;

    const std::allocator<CharT> alloc;

    {
      W w;
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(w.rdbuf()) == 0);
#endif
      {
        OS os = {w, alloc};
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
        assert(std::__wrapped_streambuf_mutex::__instance().__get_count(w.rdbuf()) == 1);
#endif
        assert(os.get_wrapped() == w.rdbuf());
        assert(os.rdbuf()->get_wrapped() == w.rdbuf());
        assert(os.rdbuf()->get_allocator() == alloc);
      }
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(w.rdbuf()) == 0);
#endif
    }
  }
  {
    using OS = std::basic_osyncstream<CharT, constexpr_char_traits<CharT>>;
    using W  = std::basic_ostringstream<CharT, constexpr_char_traits<CharT>>;

    const std::allocator<CharT> alloc;
    {
      W w;
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(w.rdbuf()) == 0);
#endif
      {
        OS os = {w.rdbuf(), alloc};
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
        assert(std::__wrapped_streambuf_mutex::__instance().__get_count(w.rdbuf()) == 1);
#endif
        assert(os.get_wrapped() == w.rdbuf());
        assert(os.rdbuf()->get_wrapped() == w.rdbuf());
        assert(os.rdbuf()->get_allocator() == alloc);
      }
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(w.rdbuf()) == 0);
#endif
    }
  }

  {
    using OS = std::basic_osyncstream<CharT, constexpr_char_traits<CharT>, test_allocator<CharT>>;
    using W  = std::basic_ostringstream<CharT, constexpr_char_traits<CharT>, test_allocator<CharT>>;

    const test_allocator<CharT> alloc;
    {
      W w;
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(w.rdbuf()) == 0);
#endif
      {
        OS os = {w.rdbuf(), alloc};
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
        assert(std::__wrapped_streambuf_mutex::__instance().__get_count(w.rdbuf()) == 1);
#endif
        assert(os.get_wrapped() == w.rdbuf());
        assert(os.rdbuf()->get_wrapped() == w.rdbuf());
        assert(os.rdbuf()->get_allocator() == alloc);
      }
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(w.rdbuf()) == 0);
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
