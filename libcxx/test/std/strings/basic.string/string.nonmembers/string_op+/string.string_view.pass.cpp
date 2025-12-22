//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=9000000

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

template <typename CharT, class TraitsT = std::char_traits<CharT>>
class ConvertibleToStringView {
public:
  constexpr explicit ConvertibleToStringView(const CharT* cs) : cs_{cs} {}

  constexpr operator std::basic_string_view<CharT, TraitsT>() { return std::basic_string_view<CharT, TraitsT>(cs_); }
  constexpr operator std::basic_string_view<CharT, TraitsT>() const {
    return std::basic_string_view<CharT, TraitsT>(cs_);
  }

private:
  const CharT* cs_;
};

static_assert(std::constructible_from<std::basic_string_view<char>, const ConvertibleToStringView<char>>);
static_assert(std::convertible_to<const ConvertibleToStringView<char>, std::basic_string_view<char>>);

static_assert(std::constructible_from<std::basic_string_view<char>, ConvertibleToStringView<char>>);
static_assert(std::convertible_to<ConvertibleToStringView<char>, std::basic_string_view<char>>);

#define CS(S) MAKE_CSTRING(CharT, S)

template <template <typename, typename> typename StringViewT, typename CharT, typename TraitsT, typename AllocT>
constexpr void test(const CharT* x, const CharT* y, const CharT* expected) {
  AllocT allocator;

  // string& + string_view
  {
    std::basic_string<CharT, TraitsT, AllocT> st{x, allocator};
    StringViewT<CharT, TraitsT> sv{y};

    std::same_as<std::basic_string<CharT, TraitsT, AllocT>> decltype(auto) result = st + sv;
    assert(result == expected);
    assert(result.get_allocator() == allocator);
    LIBCPP_ASSERT(is_string_asan_correct(st + sv));
  }
  // const string& + string_view
  {
    const std::basic_string<CharT, TraitsT, AllocT> st{x, allocator};
    StringViewT<CharT, TraitsT> sv{y};

    std::same_as<std::basic_string<CharT, TraitsT, AllocT>> decltype(auto) result = st + sv;
    assert(result == expected);
    assert(result.get_allocator() == allocator);
    LIBCPP_ASSERT(is_string_asan_correct(st + sv));
  }
  // string&& + string_view
  {
    std::basic_string<CharT, TraitsT, AllocT> st{x, allocator};
    StringViewT<CharT, TraitsT> sv{y};

    std::same_as<std::basic_string<CharT, TraitsT, AllocT>> decltype(auto) result = std::move(st) + sv;
    assert(result == expected);
    assert(result.get_allocator() == allocator);
    LIBCPP_ASSERT(is_string_asan_correct(std::move(st) + sv));
  }
  // string_view + string&
  {
    StringViewT<CharT, TraitsT> sv{x};
    std::basic_string<CharT, TraitsT, AllocT> st{y, allocator};

    std::same_as<std::basic_string<CharT, TraitsT, AllocT>> decltype(auto) result = sv + st;
    assert(result == expected);
    assert(result.get_allocator() == allocator);
    LIBCPP_ASSERT(is_string_asan_correct(sv + st));
  }
  // string_view + const string&
  {
    StringViewT<CharT, TraitsT> sv{x};
    const std::basic_string<CharT, TraitsT, AllocT> st{y, allocator};

    std::same_as<std::basic_string<CharT, TraitsT, AllocT>> decltype(auto) result = sv + st;
    assert(result == expected);
    assert(result.get_allocator() == allocator);
    LIBCPP_ASSERT(is_string_asan_correct(sv + st));
  }
  // string_view + string&&
  {
    // TODO: Remove workaround once https://llvm.org/PR92382 is fixed.
    // Create a `basic_string` to workaround clang bug:
    // https://llvm.org/PR92382
    // Comparison between pointers to a string literal and some other object results in constant evaluation failure.
    if constexpr (std::same_as<StringViewT<CharT, TraitsT>, std::basic_string_view<CharT, TraitsT>>) {
      std::basic_string<CharT, TraitsT, AllocT> st_{x, allocator};
      StringViewT<CharT, TraitsT> sv{st_};
      std::basic_string<CharT, TraitsT, AllocT> st{y, allocator};

      std::same_as<std::basic_string<CharT, TraitsT, AllocT>> decltype(auto) result = sv + std::move(st);
      assert(result == expected);
      assert(result.get_allocator() == allocator);
      LIBCPP_ASSERT(is_string_asan_correct(sv + std::move(st)));
    }
  }
}

template <template <typename, typename> typename StringViewT,
          typename CharT,
          typename TraitsT,
          typename AllocT = std::allocator<CharT>>
constexpr void test() {
  // Concatenate with an empty `string`/`string_view`
  test<StringViewT, CharT, TraitsT, AllocT>(CS(""), CS(""), CS(""));
  test<StringViewT, CharT, TraitsT, AllocT>(CS(""), CS("short"), CS("short"));
  test<StringViewT, CharT, TraitsT, AllocT>(CS(""), CS("not so short"), CS("not so short"));
  test<StringViewT, CharT, TraitsT, AllocT>(
      CS(""), CS("this is a much longer string"), CS("this is a much longer string"));

  test<StringViewT, CharT, TraitsT, AllocT>(CS(""), CS(""), CS(""));
  test<StringViewT, CharT, TraitsT, AllocT>(CS("short"), CS(""), CS("short"));
  test<StringViewT, CharT, TraitsT, AllocT>(CS("not so short"), CS(""), CS("not so short"));
  test<StringViewT, CharT, TraitsT, AllocT>(
      CS("this is a much longer string"), CS(""), CS("this is a much longer string"));

  // Non empty
  test<StringViewT, CharT, TraitsT, AllocT>(CS("B"), CS("D"), CS("BD"));
  test<StringViewT, CharT, TraitsT, AllocT>(CS("zmt94"), CS("+hkt82"), CS("zmt94+hkt82"));
  test<StringViewT, CharT, TraitsT, AllocT>(CS("not so short"), CS("+is not bad"), CS("not so short+is not bad"));
  test<StringViewT, CharT, TraitsT, AllocT>(
      CS("this is a much longer string"),
      CS("+which is so much better"),
      CS("this is a much longer string+which is so much better"));
}

template <template <typename, typename> typename StringViewT, typename CharT>
constexpr bool test() {
  test<StringViewT, CharT, std::char_traits<CharT>>();
  test<StringViewT, CharT, std::char_traits<CharT>, min_allocator<CharT>>();
  test<StringViewT, CharT, std::char_traits<CharT>, safe_allocator<CharT>>();
  test<StringViewT, CharT, std::char_traits<CharT>, test_allocator<CharT>>();

  test<StringViewT, CharT, constexpr_char_traits<CharT>>();
  test<StringViewT, CharT, constexpr_char_traits<CharT>, min_allocator<CharT>>();
  test<StringViewT, CharT, constexpr_char_traits<CharT>, safe_allocator<CharT>>();
  test<StringViewT, CharT, constexpr_char_traits<CharT>, test_allocator<CharT>>();

  return true;
}

int main(int, char**) {
  // std::basic_string_view
  test<std::basic_string_view, char>();
  static_assert(test<std::basic_string_view, char>());
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<std::basic_string_view, wchar_t>();
  static_assert(test<std::basic_string_view, wchar_t>());
#endif
#ifndef TEST_HAS_NO_CHAR8_T
  test<std::basic_string_view, char8_t>();
  static_assert(test<std::basic_string_view, char8_t>());
#endif
  test<std::basic_string_view, char16_t>();
  static_assert(test<std::basic_string_view, char16_t>());
  test<std::basic_string_view, char32_t>();
  static_assert(test<std::basic_string_view, char32_t>());

  // ConvertibleToStringView
  test<ConvertibleToStringView, char>();
  static_assert(test<ConvertibleToStringView, char>());
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<ConvertibleToStringView, wchar_t>();
  static_assert(test<ConvertibleToStringView, wchar_t>());
#endif
#ifndef TEST_HAS_NO_CHAR8_T
  test<ConvertibleToStringView, char8_t>();
  static_assert(test<ConvertibleToStringView, char8_t>());
#endif
  test<ConvertibleToStringView, char16_t>();
  static_assert(test<ConvertibleToStringView, char16_t>());
  test<ConvertibleToStringView, char32_t>();
  static_assert(test<ConvertibleToStringView, char32_t>());

  return 0;
}
