// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TEXT_ENCODING_TEXT_ENCODING_H
#define _LIBCPP___TEXT_ENCODING_TEXT_ENCODING_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__functional/hash.h>
#include <__ranges/enable_borrowed_range.h>
#include <__text_encoding/text_encoding_rep.h>
#include <cstdint>
#include <string_view>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26

_LIBCPP_BEGIN_NAMESPACE_STD

struct text_encoding {
  static constexpr size_t max_name_length = 63;
  using id                                = std::__text_encoding_rep::__id;
  using enum id;

  _LIBCPP_HIDE_FROM_ABI constexpr text_encoding() = default;
  _LIBCPP_HIDE_FROM_ABI constexpr explicit text_encoding(string_view __enc) noexcept : __rep_(__enc) {}
  _LIBCPP_HIDE_FROM_ABI constexpr text_encoding(id __i) noexcept : __rep_(__i) {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr id mib() const noexcept { return __rep_.__mib(); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr const char* name() const noexcept { return __rep_.__name(); }

  using aliases_view = __text_encoding_rep::__aliases_view;
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr aliases_view aliases() const { return __rep_.__aliases(); }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const text_encoding& __a, const text_encoding& __b) noexcept {
    return __a.__rep_ == __b.__rep_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const text_encoding& __encoding, id __i) noexcept {
    return __encoding.mib() == __i;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static consteval text_encoding literal() noexcept {
    // TODO: Remove this branch once we have __GNUC_EXECUTION_CHARSET_NAME or __clang_literal_encoding__ unconditionally
#  ifdef __GNUC_EXECUTION_CHARSET_NAME
    return text_encoding(__GNUC_EXECUTION_CHARSET_NAME);
#  elif defined(__clang_literal_encoding__)
    return text_encoding(__clang_literal_encoding__);
#  else
    return text_encoding();
#  endif
  }

#  if _LIBCPP_HAS_LOCALIZATION
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static text_encoding environment() {
    return text_encoding(__text_encoding_rep::__environment());
  }

  template <id __i>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static bool environment_is() {
    return __text_encoding_rep::__environment_is<__i>();
  }
#  else
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static text_encoding environment() = delete;
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static bool environment_is()       = delete;
#  endif // _LIBCPP_HAS_LOCALIZATION

private:
  friend class locale;

  _LIBCPP_HIDE_FROM_ABI text_encoding(__text_encoding_rep __rep) : __rep_(__rep) {}
  const __text_encoding_rep __rep_;
};

template <>
struct hash<text_encoding> {
  size_t operator()(const __text_encoding_rep& __enc) const noexcept {
    return std::hash<__text_encoding_rep::__id>()(__enc.__mib());
  }
};

template <>
inline constexpr bool ranges::enable_borrowed_range<text_encoding::aliases_view> = true;
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_POP_MACROS

#endif // _LIBCPP___TEXT_ENCODING_TEXT_ENCODING_H
