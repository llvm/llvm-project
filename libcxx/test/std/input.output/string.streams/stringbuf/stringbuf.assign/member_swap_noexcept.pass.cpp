//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_stringbuf

// void swap(basic_stringbuf& rhs)
//   noexcept(allocator_traits<allocator_type>::propagate_on_container_swap::value ||
//            allocator_traits<allocator_type>::is_always_equal::value);

#include <sstream>
#include <cassert>

#include "test_macros.h"

template <class T>
struct test_alloc {
  using value_type = T;

  [[nodiscard]] constexpr T* allocate(std::size_t) { return nullptr; }
  void deallocate(void*, unsigned) {}
};

template <class T>
struct test_alloc_propagate_on_container_swap : test_alloc<T> {
  using propagate_on_container_swap = std::true_type;
};

template <class T>
struct test_alloc_is_always_equal : test_alloc<T> {
  using is_always_equal = std::true_type;
};

template <class T>
struct test_alloc_propagate_on_container_swap_is_always_equal : test_alloc<T> {
  using propagate_on_container_swap = std::true_type;
  using is_always_equal             = std::true_type;
};

template <class T>
struct test_alloc_not_empty : test_alloc<T> {
  bool dummy;
};

template <class T>
struct test_alloc_propagate_on_container_swap_not_empty : test_alloc<T> {
  using propagate_on_container_swap = std::true_type;
  bool dummy;
};

template <class CharT>
static void test() {
  {
    std::basic_stringbuf<CharT, std::char_traits<CharT>, test_alloc<CharT>> buf1;
    std::basic_stringbuf<CharT, std::char_traits<CharT>, test_alloc<CharT>> buf;
    static_assert(noexcept(buf.swap(buf1)));
  }
  {
    std::basic_stringbuf<CharT, std::char_traits<CharT>, test_alloc_propagate_on_container_swap<CharT>> buf1;
    std::basic_stringbuf<CharT, std::char_traits<CharT>, test_alloc_propagate_on_container_swap<CharT>> buf;
    static_assert(noexcept(buf.swap(buf1)));
  }
  {
    std::basic_stringbuf<CharT, std::char_traits<CharT>, test_alloc_is_always_equal<CharT>> buf1;
    std::basic_stringbuf<CharT, std::char_traits<CharT>, test_alloc_is_always_equal<CharT>> buf;
    static_assert(noexcept(buf.swap(buf1)));
  }
  {
    std::basic_stringbuf<CharT, std::char_traits<CharT>, test_alloc_propagate_on_container_swap_is_always_equal<CharT>>
        buf1;
    std::basic_stringbuf<CharT, std::char_traits<CharT>, test_alloc_propagate_on_container_swap_is_always_equal<CharT>>
        buf;
    static_assert(noexcept(buf.swap(buf1)));
  }
  {
    std::basic_stringbuf<CharT, std::char_traits<CharT>, test_alloc_not_empty<CharT>> buf1;
    std::basic_stringbuf<CharT, std::char_traits<CharT>, test_alloc_not_empty<CharT>> buf;
    static_assert(!noexcept(buf.swap(buf1)));
  }
  {
    std::basic_stringbuf<CharT, std::char_traits<CharT>, test_alloc_propagate_on_container_swap_not_empty<CharT>> buf1;
    std::basic_stringbuf<CharT, std::char_traits<CharT>, test_alloc_propagate_on_container_swap_not_empty<CharT>> buf;
    static_assert(noexcept(buf.swap(buf1)));
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  return 0;
}
