//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <string>

// [string.op.plus]
//
// template<class charT, class traits, class Allocator>
//   constexpr basic_string<charT, traits, Allocator>
//     operator+(const basic_string<charT, traits, Allocator>& lhs,
//               type_identity_t<basic_string_view<charT, traits>> rhs);                           // Since C++26
// template<class charT, class traits, class Allocator>
//   constexpr basic_string<charT, traits, Allocator>
//     operator+(basic_string<charT, traits, Allocator>&& lhs,
//               type_identity_t<basic_string_view<charT, traits>> rhs);                           // Since C++26
// template<class charT, class traits, class Allocator>
//   constexpr basic_string<charT, traits, Allocator>
//     operator+(type_identity_t<basic_string_view<charT, traits>> lhs,
//               const basic_string<charT, traits, Allocator>& rhs);                               // Since C++26
// template<class charT, class traits, class Allocator>
//   constexpr basic_string<charT, traits, Allocator>
//     operator+(type_identity_t<basic_string_view<charT, traits>> lhs,
//               basic_string<charT, traits, Allocator>&& rhs);                                    // Since C++26

#include <cassert>
#include <concepts>
#include <string>
#include <utility>

#include "asan_testing.h"
#include "constexpr_char_traits.h"
#include "make_string.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

#define CS(S) MAKE_CSTRING(CharT, S)
#define ST(S, a) std::basic_string<CharT, TraitsT, AllocT>(MAKE_CSTRING(CharT, S), MKSTR_LEN(CharT, S), a)
#define SV(S) std::basic_string_view<CharT, TraitsT>(MAKE_CSTRING(CharT, S), MKSTR_LEN(CharT, S))

template <typename CharT, typename TraitsT, typename AllocT>
constexpr void test(const CharT* x, const CharT* y, const CharT* expected) {
  AllocT allocator;

  // string& + string_view
  {
    std::basic_string<CharT, TraitsT, AllocT> st{x, allocator};
    std::basic_string_view<CharT, TraitsT> sv{y};

    std::same_as<std::basic_string<CharT, TraitsT, AllocT>> decltype(auto) result = st + sv;
    assert(result == expected);
    assert(result.get_allocator() == allocator);
    LIBCPP_ASSERT(is_string_asan_correct(st + sv));
  }
  // const string& + string_view
  {
    const std::basic_string<CharT, TraitsT, AllocT> st{x, allocator};
    std::basic_string_view<CharT, TraitsT> sv{y};

    std::same_as<std::basic_string<CharT, TraitsT, AllocT>> decltype(auto) result = st + sv;
    assert(result == expected);
    assert(result.get_allocator() == allocator);
    LIBCPP_ASSERT(is_string_asan_correct(st + sv));
  }
  // string&& + string_view
  {
    std::basic_string<CharT, TraitsT, AllocT> st{x, allocator};
    std::basic_string_view<CharT, TraitsT> sv{y};

    std::same_as<std::basic_string<CharT, TraitsT, AllocT>> decltype(auto) result = std::move(st) + sv;
    assert(result == expected);
    assert(result.get_allocator() == allocator);
    LIBCPP_ASSERT(is_string_asan_correct(st + sv));
  }
  // string_view + string&
  {
    std::basic_string_view<CharT, TraitsT> sv{x};
    std::basic_string<CharT, TraitsT, AllocT> st{y, allocator};

    std::same_as<std::basic_string<CharT, TraitsT, AllocT>> decltype(auto) result = sv + st;
    assert(result == expected);
    assert(result.get_allocator() == allocator);
    LIBCPP_ASSERT(is_string_asan_correct(sv + st));
  }
  // string_view + const string&
  {
    std::basic_string_view<CharT, TraitsT> sv{x};
    const std::basic_string<CharT, TraitsT, AllocT> st{y, allocator};

    std::same_as<std::basic_string<CharT, TraitsT, AllocT>> decltype(auto) result = sv + st;
    assert(result == expected);
    assert(result.get_allocator() == allocator);
    LIBCPP_ASSERT(is_string_asan_correct(sv + st));
  }
  // string_view + string&&
  {
    std::basic_string<CharT, TraitsT, AllocT> st_{x, allocator};
    std::basic_string_view<CharT, TraitsT> sv{st_};
    std::basic_string<CharT, TraitsT, AllocT> st{y, allocator};

    std::same_as<std::basic_string<CharT, TraitsT, AllocT>> decltype(auto) result = sv + std::move(st);
    assert(result == expected);
    assert(result.get_allocator() == allocator);
    LIBCPP_ASSERT(is_string_asan_correct(sv + st));
  }
}

template <typename CharT, typename TraitsT, typename AllocT = std::allocator<CharT>>
constexpr void test() {
  // Concatenate with an empty `string`/`string_view`
  test<CharT, TraitsT, AllocT>(CS(""), CS(""), CS(""));
  test<CharT, TraitsT, AllocT>(CS(""), CS("short"), CS("short"));
  test<CharT, TraitsT, AllocT>(CS(""), CS("not so short"), CS("not so short"));
  test<CharT, TraitsT, AllocT>(CS(""), CS("this is a much longer string"), CS("this is a much longer string"));

  test<CharT, TraitsT, AllocT>(CS(""), CS(""), CS(""));
  test<CharT, TraitsT, AllocT>(CS("short"), CS(""), CS("short"));
  test<CharT, TraitsT, AllocT>(CS("not so short"), CS(""), CS("not so short"));
  test<CharT, TraitsT, AllocT>(CS("this is a much longer string"), CS(""), CS("this is a much longer string"));

  // Non empty
  test<CharT, TraitsT, AllocT>(CS("B"), CS("D"), CS("BD"));
  test<CharT, TraitsT, AllocT>(CS("zmt94"), CS("+hkt82"), CS("zmt94+hkt82"));
  test<CharT, TraitsT, AllocT>(CS("not so short"), CS("+is not bad"), CS("not so short+is not bad"));
  test<CharT, TraitsT, AllocT>(
      CS("this is a much longer string"),
      CS("+which is so much better"),
      CS("this is a much longer string+which is so much better"));
}

template <typename CharT>
constexpr bool test() {
  test<CharT, std::char_traits<CharT>>();
  test<CharT, std::char_traits<CharT>, min_allocator<CharT>>();
  test<CharT, std::char_traits<CharT>, safe_allocator<CharT>>();
  test<CharT, std::char_traits<CharT>, test_allocator<CharT>>();

  test<CharT, constexpr_char_traits<CharT>>();
  test<CharT, constexpr_char_traits<CharT>, min_allocator<CharT>>();
  test<CharT, constexpr_char_traits<CharT>, safe_allocator<CharT>>();
  test<CharT, constexpr_char_traits<CharT>, test_allocator<CharT>>();

  return true;
}

int main(int, char**) {
  test<char>();
  static_assert(test<char>());
  #ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
    static_assert(test<wchar_t>());
  #endif
  #ifndef TEST_HAS_NO_CHAR8_T
  test<char8_t>();
    static_assert(test<char8_t>());
  #endif
  test<char16_t>();
    static_assert(test<char16_t>());
  test<char32_t>();
    static_assert(test<char32_t>());

  return 0;
}
