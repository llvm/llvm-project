//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBCPP_TEST_SUPPORT_PARSE_INTEGER_H
#define LIBCPP_TEST_SUPPORT_PARSE_INTEGER_H

#include <charconv>
#include <cstdlib>
#include <string>
#include <system_error>
#include <type_traits>

#include "test_macros.h"

namespace detail {
template <class T>
struct parse_integer_impl;

#if TEST_STD_VER >= 23
template <class I, class CharT>
consteval I consteval_parse_integer(const std::basic_string<CharT>& str) {
  I n;
  auto [ptr, ec] = [&n, &str] {
    if constexpr (std::is_same_v<CharT, char>) {
      return std::from_chars(str.data(), str.data() + str.size(), n);
    } else {
      std::string s;
      for (auto c : str)
        s.push_back(static_cast<char>(c));
      return std::from_chars(s.data(), s.data() + s.size(), n);
    }
  }();
  if (ec != std::errc{})
    std::abort();
  return n;
}
#endif

template <>
struct parse_integer_impl<int> {
  template <class CharT>
  TEST_CONSTEXPR_CXX23 int operator()(std::basic_string<CharT> const& str) const {
#if TEST_STD_VER >= 23
    if consteval {
      return consteval_parse_integer<int>(str);
    }
#endif
    return std::stoi(str);
  }
};

template <>
struct parse_integer_impl<long> {
  template <class CharT>
  TEST_CONSTEXPR_CXX23 long operator()(std::basic_string<CharT> const& str) const {
#if TEST_STD_VER >= 23
    if consteval {
      return consteval_parse_integer<long>(str);
    }
#endif
    return std::stol(str);
  }
};

template <>
struct parse_integer_impl<long long> {
  template <class CharT>
  TEST_CONSTEXPR_CXX23 long long operator()(std::basic_string<CharT> const& str) const {
#if TEST_STD_VER >= 23
    if consteval {
      return consteval_parse_integer<long long>(str);
    }
#endif
    return std::stoll(str);
  }
};

template <>
struct parse_integer_impl<unsigned int> {
  template <class CharT>
  TEST_CONSTEXPR_CXX23 unsigned int operator()(std::basic_string<CharT> const& str) const {
#if TEST_STD_VER >= 23
    if consteval {
      return consteval_parse_integer<unsigned int>(str);
    }
#endif
    return std::stoul(str);
  }
};

template <>
struct parse_integer_impl<unsigned long> {
  template <class CharT>
  TEST_CONSTEXPR_CXX23 unsigned long operator()(std::basic_string<CharT> const& str) const {
#if TEST_STD_VER >= 23
    if consteval {
      return consteval_parse_integer<unsigned long>(str);
    }
#endif
    return std::stoul(str);
  }
};

template <>
struct parse_integer_impl<unsigned long long> {
  template <class CharT>
  TEST_CONSTEXPR_CXX23 unsigned long long operator()(std::basic_string<CharT> const& str) const {
#if TEST_STD_VER >= 23
    if consteval {
      return consteval_parse_integer<unsigned long long>(str);
    }
#endif
    return std::stoull(str);
  }
};
} // namespace detail

template <class T, class CharT>
TEST_CONSTEXPR_CXX23 T parse_integer(std::basic_string<CharT> const& str) {
  return detail::parse_integer_impl<T>()(str);
}

#endif // LIBCPP_TEST_SUPPORT_PARSE_INTEGER_H
