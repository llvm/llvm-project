//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "libherbceptions.h"
#include <herbceptions/error>

namespace std::error_domains::__herbceptions_detail {

inline constexpr ::std::io_scatter_t
__to_u8scatter_from_posix_errc(int __eno) noexcept {
  using ::std::error_domains::__herbceptions_detail::__tsc;
  switch (__eno) {
#include "posix_table.hpp"
  }
}

inline constexpr ::std::io_scatter_t
__to_u8scatter_from_parse_errc(::std::uint_least32_t __eno) noexcept {
  using ::std::error_domains::__herbceptions_detail::__tsc;
  switch (__eno) {
#include "parse_table.hpp"
  }
}

inline constexpr ::std::io_scatter_t
__to_u8scatter_from_cmath_errc(::std::uint_least32_t __eno) noexcept {
  using ::std::error_domains::__herbceptions_detail::__tsc;
  switch (__eno) {
#include "cmath_table.hpp"
  }
}

inline constexpr bool __simple_query_information_enable_wine_errc{
#if defined(_WIN32) || defined(__CYGWIN__)
    true
#else
    false
#endif
};

inline constexpr ::std::io_scatter_t
__to_u8scatter_from_wine_errc(::std::uint_least32_t __eno) noexcept {
  using ::std::error_domains::__herbceptions_detail::__tsc;
  switch (__eno) {
#include "wine_table.hpp"
  }
}

inline constexpr ::std::io_scatter_t
__to_u8scatter_from_errcs(::std::uint_least32_t __eno,
                          ::std::uint_least8_t __which_errc) noexcept {
  switch (__which_errc) {
  case 0:
    return __to_u8scatter_from_posix_errc(static_cast<int>(__eno));
  case 1: // cmath
    return __to_u8scatter_from_cmath_errc(__eno);
  case 2: // parse
    return __to_u8scatter_from_parse_errc(__eno);
  case 3: // wine
    if constexpr (__simple_query_information_enable_wine_errc) {
      return __to_u8scatter_from_parse_errc(__eno);
    }
    [[fallthrough]];
  default:
    __builtin_trap();
  };
  return {};
}

inline constexpr ::std::size_t
__compute_max_buffer_size_for_simple_query() noexcept {
  ::std::size_t mxsz{POSIX_ERRC_MAX_SIZE};
  if (mxsz < PARSE_ERRC_MAX_SIZE) {
    mxsz = PARSE_ERRC_MAX_SIZE;
  }
  if (mxsz < CMATH_ERRC_MAX_SIZE) {
    mxsz = CMATH_ERRC_MAX_SIZE;
  }
#ifdef WINE_ERRC_MAX_SIZE
  if (mxsz < WINE_ERRC_MAX_SIZE) {
    mxsz = WINE_ERRC_MAX_SIZE;
  }
#endif
  return mxsz;
}

inline constexpr ::std::io_scatter_t
__posix_name_message_range(::std::error_reporter_encoding encoding,
                           ::std::size_t startpos, ::std::size_t n) noexcept {
  switch (encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
  case ::std::error_reporter_encoding::utfebcdic: {
    return {&startpos["\xAD\x97\x96\xA2\x89\xA7\xBD"], n};
  }
#endif
  case ::std::error_reporter_encoding::utf16: {
    return {&startpos[u"[posix]"], n * sizeof(char16_t)};
  }
  case ::std::error_reporter_encoding::utf32: {
    return {&startpos[U"[posix]"], n * sizeof(char32_t)};
  }
  default: {
    return {&startpos[u8"[posix]"], n};
  }
  }
}

inline constexpr ::std::io_scatter_t
__parse_name_message_range(::std::error_reporter_encoding encoding,
                           ::std::size_t startpos, ::std::size_t n) noexcept {
  switch (encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
  case ::std::error_reporter_encoding::utfebcdic: {
    return {&startpos["\xD1\x97\x99\xA2\x85\xA7\xBD"], n};
  }
#endif
  case ::std::error_reporter_encoding::utf16: {
    return {&startpos[u"[parse]"], n * sizeof(char16_t)};
  }
  case ::std::error_reporter_encoding::utf32: {
    return {&startpos[U"[parse]"], n * sizeof(char32_t)};
  }
  default: {
    return {&startpos[u8"[parse]"], n};
  }
  }
}

inline constexpr ::std::io_scatter_t
__cmath_name_message_range(::std::error_reporter_encoding encoding,
                           ::std::size_t startpos, ::std::size_t n) noexcept {
  switch (encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
  case ::std::error_reporter_encoding::utfebcdic: {
    return {&startpos["\xC3\x89\x96\xA3\x89\xA7\xBD"], n};
  }
#endif
  case ::std::error_reporter_encoding::utf16: {
    return {&startpos[u"[cmath]"], n * sizeof(char16_t)};
  }
  case ::std::error_reporter_encoding::utf32: {
    return {&startpos[U"[cmath]"], n * sizeof(char32_t)};
  }
  default: {
    return {&startpos[u8"[cmath]"], n};
  }
  }
}

inline constexpr ::std::io_scatter_t
__wine_name_message_range(::std::error_reporter_encoding encoding,
                          ::std::size_t startpos, ::std::size_t n) noexcept {
  switch (encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
  case ::std::error_reporter_encoding::utfebcdic: {
    return {&startpos["\xE2\x97\xA2\xA2\x85\xA7\xBD"], n};
  }
#endif
  case ::std::error_reporter_encoding::utf16: {
    return {&startpos[u"[wine]"], n * sizeof(char16_t)};
  }
  case ::std::error_reporter_encoding::utf32: {
    return {&startpos[U"[wine]"], n * sizeof(char32_t)};
  }
  default: {
    return {&startpos[u8"[wine]"], n};
  }
  }
}
inline constexpr ::std::io_scatter_t __simple_query_information_common_name(
    ::std::error_reporter_encoding encoding,
    ::std::uint_least8_t which_errc) noexcept {
  switch (which_errc) {
  case 0:
    /*
    posix
    */
    return __posix_name_message_range(encoding, 1u, 5u);
  case 1:
    /*
    cmath
    */
    return __cmath_name_message_range(encoding, 1u, 5u);
  case 2:
    /*
    parse
    */
    return __parse_name_message_range(encoding, 1u, 5u);
  case 3:
    /*
    wine
    */
    if constexpr (__simple_query_information_enable_wine_errc) {
      return __wine_name_message_range(encoding, 1u, 4u);
    }
    [[fallthrough]];
  default:
    __builtin_trap();
  };
}

inline constexpr ::std::io_scatter_t __simple_query_information_common_message(
    ::std::error_reporter_encoding encoding,
    ::std::uint_least8_t which_errc) noexcept {
  switch (which_errc) {
  case 0:
    /*
    [posix]
    */
    return __posix_name_message_range(encoding, 0u, 7u);
  case 1:
    /*
    [cmath]
    */
    return __cmath_name_message_range(encoding, 0u, 7u);
  case 2:
    /*
    [parse]
    */
    return __parse_name_message_range(encoding, 0u, 7u);
  case 3:
    /*
    [wine]
    */
    if constexpr (__simple_query_information_enable_wine_errc) {
      return __wine_name_message_range(encoding, 0u, 6u);
    }
    [[fallthrough]];
  default:
    __builtin_trap();
  };
}

/*
Writes "(<decimal code>)" for the requested encoding into __numbuf and
returns it as a scatter. The buffer must be at least
__format_decimal_value_max_size_with_brackets<::std::uint_least32_t>
code units wide, each of the largest supported character size.
*/
inline constexpr ::std::io_scatter_t
__simple_query_information_common_code(::std::error_reporter_encoding encoding,
                                       ::std::uint_least32_t __code,
                                       char unsigned *__numbuf) noexcept {
  switch (encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
  case ::std::error_reporter_encoding::utfebcdic: {
    auto *__dest{
        ::std::error_domains::__herbceptions_detail::
            __format_decimal_value_full_with_bracket<true, char unsigned>(
                __numbuf, __code)};
    return {__numbuf, static_cast<::std::size_t>(__dest - __numbuf)};
  }
#endif
  case ::std::error_reporter_encoding::utf16: {
    using __char16_may_alias_ptr
#if __has_cpp_attribute(__gnu__::__may_alias__)
        [[__gnu__::__may_alias__]]
#endif
        = char16_t *;
    auto *__dest{
        ::std::error_domains::__herbceptions_detail::
            __format_decimal_value_full_with_bracket<false, char16_t>(
                reinterpret_cast<__char16_may_alias_ptr>(__numbuf), __code)};
    return {__numbuf,
            static_cast<::std::size_t>(
                reinterpret_cast<char unsigned *>(__dest) - __numbuf)};
  }
  case ::std::error_reporter_encoding::utf32: {
    using __char32_may_alias_ptr
#if __has_cpp_attribute(__gnu__::__may_alias__)
        [[__gnu__::__may_alias__]]
#endif
        = char32_t *;
    auto *__dest{
        ::std::error_domains::__herbceptions_detail::
            __format_decimal_value_full_with_bracket<false, char32_t>(
                reinterpret_cast<__char32_may_alias_ptr>(__numbuf), __code)};
    return {__numbuf,
            static_cast<::std::size_t>(
                reinterpret_cast<char unsigned *>(__dest) - __numbuf)};
  }
  default: {
    auto *__dest{
        ::std::error_domains::__herbceptions_detail::
            __format_decimal_value_full_with_bracket<false, char unsigned>(
                __numbuf, __code)};
    return {__numbuf, static_cast<::std::size_t>(__dest - __numbuf)};
  }
  }
}

inline constexpr void __simple_query_information_common(
    ::std::size_t cd, ::std::error_query_information query,
    ::std::error_reporter_encoding encoding, void *cookie,
    ::std::error_reporter_io_cookie_function cookfun,
    ::std::uint_least8_t which_errc) noexcept {
  if (static_cast<::std::uint_least32_t>(
          ::std::error_query_information::name_message) <
      static_cast<::std::uint_least32_t>(query)) {
    return;
  }
  auto const __code{static_cast<::std::uint_least32_t>(cd)};
  if constexpr (::std::error_domains::__herbceptions_detail::
                    __is_freestanding_kernel_mode) {
    /*
    Freestanding kernel mode: the message tables are a waste of rom size,
    so only [domain](numeric code) is ever reported and the text is never
    touched. Only a small fixed scratch buffer is used.
    */
    ::std::io_scatter_t __scatters[2];
    ::std::size_t __scatterlen{};
    switch (query) {
    case ::std::error_query_information::name: {
      *__scatters =
          __simple_query_information_common_name(encoding, which_errc);
      __scatterlen = 1u;
      break;
    }
    case ::std::error_query_information::message:
      [[fallthrough]];
    case ::std::error_query_information::name_message: {
      alignas(char32_t) char unsigned
          __numbuf[__format_decimal_value_max_size_with_brackets<
                       ::std::uint_least32_t> *
                   sizeof(char32_t)];
      __scatterlen = 0u;
      if (::std::error_query_information::name_message == query) {
        *__scatters =
            __simple_query_information_common_message(encoding, which_errc);
        ++__scatterlen;
      }
      __scatters[__scatterlen] =
          __simple_query_information_common_code(encoding, __code, __numbuf);
      ++__scatterlen;
      break;
    }
    }
    cookfun(cookie, __scatters, __scatterlen);
  } else {
    ::std::io_scatter_t __scatters[3];
    ::std::size_t __scatterlen{};
    constexpr ::std::size_t __errno_max_bytes{
        ::std::error_domains::__herbceptions_detail::
            __compute_max_buffer_size_for_simple_query() *
        sizeof(char32_t)};
    alignas(char32_t) char unsigned __buffer[__errno_max_bytes];
    switch (query) {
    case ::std::error_query_information::name: {
      *__scatters =
          __simple_query_information_common_name(encoding, which_errc);
      __scatterlen = 1u;
      break;
    }
    case ::std::error_query_information::message:
      [[fallthrough]];
    case ::std::error_query_information::name_message: {
      alignas(char32_t) char unsigned
          __numbuf[__format_decimal_value_max_size_with_brackets<
                       ::std::uint_least32_t> *
                   sizeof(char32_t)];
      if (::std::error_query_information::name_message == query) {
        *__scatters =
            __simple_query_information_common_message(encoding, which_errc);
        ++__scatterlen;
      }
      __scatters[__scatterlen] =
          __simple_query_information_common_code(encoding, __code, __numbuf);
      ++__scatterlen;
      auto __scatter{::std::error_domains::__herbceptions_detail::
                         __to_u8scatter_from_errcs(__code, which_errc)};
      char unsigned const *__from_first{
          reinterpret_cast<char unsigned const *>(__scatter.base)};
      char unsigned const *__from_last{__from_first + __scatter.len};
      switch (encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
      case ::std::error_reporter_encoding::utfebcdic: {
        auto __dest = ::std::error_domains::__herbceptions_detail::
            __write_ebcdic_with_ascii_only_range(__from_first, __from_last,
                                                 __buffer);
        __scatters[__scatterlen] = {
            __buffer,
            static_cast<::std::size_t>(
                reinterpret_cast<char unsigned *>(__dest) - __buffer)};
        break;
      }
#endif
      case ::std::error_reporter_encoding::utf16: {
        using __char16_may_alias_ptr
#if __has_cpp_attribute(__gnu__::__may_alias__)
            [[__gnu__::__may_alias__]]
#endif
            = char16_t *;
        auto __dest = ::std::error_domains::__herbceptions_detail::
            __write_with_ascii_only_range(
                __from_first, __from_last,
                reinterpret_cast<__char16_may_alias_ptr>(__buffer));
        __scatters[__scatterlen] = {
            __buffer,
            static_cast<::std::size_t>(
                reinterpret_cast<char unsigned *>(__dest) - __buffer)};
        break;
      }
      case ::std::error_reporter_encoding::utf32: {
        using __char32_may_alias_ptr
#if __has_cpp_attribute(__gnu__::__may_alias__)
            [[__gnu__::__may_alias__]]
#endif
            = char32_t *;
        auto __dest = ::std::error_domains::__herbceptions_detail::
            __write_with_ascii_only_range(
                __from_first, __from_last,
                reinterpret_cast<__char32_may_alias_ptr>(__buffer));
        __scatters[__scatterlen] = {
            __buffer,
            static_cast<::std::size_t>(
                reinterpret_cast<char unsigned *>(__dest) - __buffer)};
        break;
      }
      default: {
        __scatters[__scatterlen] = __scatter;
        break;
      }
      }
      ++__scatterlen;
      break;
    }
    }
    cookfun(cookie, __scatters, __scatterlen);
  }
}

} // namespace std::error_domains::__herbceptions_detail
