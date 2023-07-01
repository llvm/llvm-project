//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>
// UNSUPPORTED: c++03, c++11, c++14

// template<class InputIterator,
//      class Allocator = allocator<typename iterator_traits<InputIterator>::value_type>>
//  basic_string(InputIterator, InputIterator, Allocator = Allocator())
//    -> basic_string<typename iterator_traits<InputIterator>::value_type,
//                 char_traits<typename iterator_traits<InputIterator>::value_type>,
//                 Allocator>; // constexpr since C++20
//
//  The deduction guide shall not participate in overload resolution if InputIterator
//  is a type that does not qualify as an input iterator, or if Allocator is a type
//  that does not qualify as an allocator.

#include <cassert>
#include <cstddef>
#include <iterator>
#include <string>
#include <type_traits>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

class NotAnIterator {};
using NotAnInputIterator = std::back_insert_iterator<std::basic_string<char16_t>>;

template <typename T>
struct NotAnAllocator { typedef T value_type; };

template <class Iter, class Alloc, class = void>
struct CanDeduce : std::false_type { };

template <class Iter, class Alloc>
struct CanDeduce<Iter, Alloc, decltype((void)
  std::basic_string{std::declval<Iter>(), std::declval<Iter>(), std::declval<Alloc>()}
)> : std::true_type { };

static_assert( CanDeduce<char*, std::allocator<char>>::value);
static_assert(!CanDeduce<NotAnIterator, std::allocator<char>>::value);
static_assert(!CanDeduce<NotAnInputIterator, std::allocator<char16_t>>::value);
static_assert(!CanDeduce<char*, NotAnAllocator<char>>::value);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert( CanDeduce<wchar_t*, std::allocator<wchar_t>>::value);
static_assert(!CanDeduce<wchar_t const*, NotAnAllocator<wchar_t>>::value);
#endif

TEST_CONSTEXPR_CXX20 bool test() {
  {
    const char* s = "12345678901234";
    std::basic_string s1(s, s+10);  // Can't use {} here
    using S = decltype(s1); // what type did we get?
    static_assert(std::is_same_v<S::value_type,                      char>,  "");
    static_assert(std::is_same_v<S::traits_type,    std::char_traits<char>>, "");
    static_assert(std::is_same_v<S::allocator_type,   std::allocator<char>>, "");
    assert(s1.size() == 10);
    assert(s1.compare(0, s1.size(), s, s1.size()) == 0);
  }
  {
    const char* s = "12345678901234";
    std::basic_string s1{s, s+10, std::allocator<char>{}};
    using S = decltype(s1); // what type did we get?
    static_assert(std::is_same_v<S::value_type,                      char>,  "");
    static_assert(std::is_same_v<S::traits_type,    std::char_traits<char>>, "");
    static_assert(std::is_same_v<S::allocator_type,   std::allocator<char>>, "");
    assert(s1.size() == 10);
    assert(s1.compare(0, s1.size(), s, s1.size()) == 0);
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    const wchar_t* s = L"12345678901234";
    std::basic_string s1{s, s+10, test_allocator<wchar_t>{}};
    using S = decltype(s1); // what type did we get?
    static_assert(std::is_same_v<S::value_type,                      wchar_t>,  "");
    static_assert(std::is_same_v<S::traits_type,    std::char_traits<wchar_t>>, "");
    static_assert(std::is_same_v<S::allocator_type,   test_allocator<wchar_t>>, "");
    assert(s1.size() == 10);
    assert(s1.compare(0, s1.size(), s, s1.size()) == 0);
  }
#endif
  {
    const char16_t* s = u"12345678901234";
    std::basic_string s1{s, s+10, min_allocator<char16_t>{}};
    using S = decltype(s1); // what type did we get?
    static_assert(std::is_same_v<S::value_type,                      char16_t>,  "");
    static_assert(std::is_same_v<S::traits_type,    std::char_traits<char16_t>>, "");
    static_assert(std::is_same_v<S::allocator_type,    min_allocator<char16_t>>, "");
    assert(s1.size() == 10);
    assert(s1.compare(0, s1.size(), s, s1.size()) == 0);
  }
  {
    const char32_t* s = U"12345678901234";
    std::basic_string s1{s, s+10, explicit_allocator<char32_t>{}};
    using S = decltype(s1); // what type did we get?
    static_assert(std::is_same_v<S::value_type,                        char32_t>,  "");
    static_assert(std::is_same_v<S::traits_type,      std::char_traits<char32_t>>, "");
    static_assert(std::is_same_v<S::allocator_type, explicit_allocator<char32_t>>, "");
    assert(s1.size() == 10);
    assert(s1.compare(0, s1.size(), s, s1.size()) == 0);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
