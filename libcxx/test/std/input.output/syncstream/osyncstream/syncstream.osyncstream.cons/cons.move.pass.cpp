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

// basic_osyncstream(basic_osyncstream&&) noexcept;

// TODO Why is this noexcept?
// Does the reasoning for https://cplusplus.github.io/LWG/issue3867 not hold true here?

#include <cassert>
#include <sstream>
#include <syncstream>

#include "test_macros.h"
#include "constexpr_char_traits.h"
#include "test_allocator.h"

template <class CharT>
void test() {
  {
    using OS = std::basic_osyncstream<CharT>;
    using W  = std::basic_syncbuf<CharT>;

    const std::allocator<CharT> alloc;
    {
      OS os = {OS{nullptr, alloc}};
      assert(os.get_wrapped() == nullptr);
      assert(os.rdbuf()->get_wrapped() == nullptr);
      assert(os.rdbuf()->get_allocator() == alloc);
      ASSERT_NOEXCEPT(OS{std::move(os)});
    }
    {
      W w;
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
      {
        OS os = {OS{&w, alloc}};
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
        assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 1);
#endif
        assert(os.get_wrapped() == &w);
        assert(os.rdbuf()->get_wrapped() == &w);
        assert(os.rdbuf()->get_allocator() == alloc);
      }
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
    }
  }

  {
    using OS = std::basic_osyncstream<CharT, constexpr_char_traits<CharT>>;
    using W  = std::basic_stringbuf<CharT, constexpr_char_traits<CharT>>;

    const std::allocator<CharT> alloc;
    {
      OS os = {OS{nullptr, alloc}};
      assert(os.get_wrapped() == nullptr);
      assert(os.rdbuf()->get_wrapped() == nullptr);
      assert(os.rdbuf()->get_allocator() == alloc);
    }
    {
      W w;
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
      {
        OS os = {OS{&w, alloc}};
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
        assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 1);
#endif
        assert(os.get_wrapped() == &w);
        assert(os.rdbuf()->get_wrapped() == &w);
        assert(os.rdbuf()->get_allocator() == alloc);
      }
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
    }
  }

  {
    using OS = std::basic_osyncstream<CharT, constexpr_char_traits<CharT>, test_allocator<CharT>>;
    using W  = std::basic_stringbuf<CharT, constexpr_char_traits<CharT>, test_allocator<CharT>>;

    const test_allocator<CharT> alloc;
    {
      OS os = {OS{nullptr, alloc}};
      assert(os.get_wrapped() == nullptr);
      assert(os.rdbuf()->get_wrapped() == nullptr);
      assert(os.rdbuf()->get_allocator() == alloc);
    }
    {
      W w;
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
      assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 0);
#endif
      {
        OS os = {OS{&w, alloc}};
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_THREADS)
        assert(std::__wrapped_streambuf_mutex::__instance().__get_count(&w) == 1);
#endif
        assert(os.get_wrapped() == &w);
        assert(os.rdbuf()->get_wrapped() == &w);
        assert(os.rdbuf()->get_allocator() == alloc);
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
