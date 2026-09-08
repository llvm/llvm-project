//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
/*
Shared helpers for the herbception error-domain runtime.

These are internal to the runtime implementation (used by the per-domain
translation units) and are not part of the public herbception/error surface.
*/
#include <herbceptions/__details/macros_guard.h>
#include <herbceptions/error>
#include <limits>
#include <type_traits>

namespace std::error_domains::__herbceptions_detail {

template <typename __Ty, ::std::size_t __n>
inline constexpr ::std::io_scatter_t __tsc(__Ty const (&__arr)[__n]) noexcept {
  constexpr ::std::size_t __nm1{__n - 1u};
  return {__arr, __nm1};
}

template <typename __SrcTy, typename __DestTy>
inline constexpr __DestTy *
__write_with_ascii_only_range(__SrcTy const *__fromfirst,
                              __SrcTy const *__fromlast, __DestTy *__dest) {
  for (; __fromfirst != __fromlast; ++__fromfirst) {
    *__dest = *__fromfirst;
    ++__dest;
  }
  return __dest;
}

template <typename __SrcTy, typename __DestTy>
inline constexpr __DestTy *__write_with_ascii_only_badcode_range(
    __SrcTy const *__fromfirst, __SrcTy const *__fromlast, __DestTy *__dest) {
  for (; __fromfirst != __fromlast; ++__fromfirst) {
    __DestTy __cp{*__fromfirst};
    if (0x80 <= __cp) {
      __cp = 0xFEFF;
    }
    *__dest = __cp;
    ;
    ++__dest;
  }
  return __dest;
}

#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
#include "ascii_to_ebcdic.cpp"

template <typename __SrcTy>
inline constexpr char unsigned *
__write_ebcdic_with_ascii_only_range(__SrcTy const *__fromfirst,
                                     __SrcTy const *__fromlast,
                                     char unsigned *__dest) {
  for (; __fromfirst != __fromlast; ++__fromfirst) {
    *__dest = ::std::error_domains::__herbceptions_detail::__ascii_to_ebcdic(
        *__fromfirst);
    ++__dest;
  }
  return __dest;
}
#endif

inline char unsigned *__codecvt_write_with_encoding(
    char unsigned const *__fromfirst, char unsigned const *__fromlast,
    char unsigned *__dest, ::std::error_reporter_encoding __encoding) noexcept {
  if (__fromfirst == __fromlast) {
    return __dest;
  }
  switch (__encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
  case ::std::error_reporter_encoding::utfebcdic: {
    if constexpr (::std::error_domains::__herbceptions_detail::
                      __libherbceptions_enable_ebcdic) {
      return ::std::error_domains::__herbceptions_detail::
          __write_ebcdic_with_ascii_only_range(__fromfirst, __fromlast, __dest);
    } else {
      [[unreachable]];
    }
  }
#endif
  case ::std::error_reporter_encoding::utf32:
    using __char32_may_alias_ptr
#if __has_cpp_attribute(__gnu__::__may_alias__)
        [[__gnu__::__may_alias__]]
#endif
        = char32_t *;
    return reinterpret_cast<char unsigned *>(
        ::std::error_domains::__herbceptions_detail::
            __write_with_ascii_only_badcode_range(
                __fromfirst, __fromlast,
                reinterpret_cast<__char32_may_alias_ptr>(__dest)));
  default:
    using __char16_may_alias_ptr
#if __has_cpp_attribute(__gnu__::__may_alias__)
        [[__gnu__::__may_alias__]]
#endif
        = char16_t *;
    return reinterpret_cast<char unsigned *>(
        ::std::error_domains::__herbceptions_detail::
            __write_with_ascii_only_badcode_range(
                __fromfirst, __fromlast,
                reinterpret_cast<__char16_may_alias_ptr>(__dest)));
  }
}

inline constexpr bool __enable_message_query{
#ifdef __libherbceptions_enable_message_query
    true
#endif
};

template <char8_t __asciicp, typename __chartype>
inline constexpr __chartype __char_literal_v{__asciicp};

/*
In Freestanding Mode, we do not print message, only the code
value to avoid Heap Allocation routines and stack overflow
*/
inline constexpr bool __is_freestanding_kernel_mode{
#if __STDC_HOSTED__ == 0 || _KERNEL_MODE == 1
    true
#endif
};

template <typename T>
inline constexpr ::std::size_t __compute_format_hex_value_max_size() noexcept {
  constexpr ::std::size_t mxhex{::std::numeric_limits<T>::digits};
  return mxhex >> 2u;
}

template <typename T>
inline constexpr ::std::size_t __format_hex_value_max_size_no_sign{
    static_cast<::std::size_t>(
        (static_cast<::std::size_t>(::std::numeric_limits<T>::digits) >> 2u))};
template <typename T>
inline constexpr ::std::size_t __format_hex_value_max_size{
    ::std::error_domains::__herbceptions_detail::
        __format_hex_value_max_size_no_sign<T>};
template <typename T>
inline constexpr ::std::size_t __format_hex_value_max_size_with_brackets{
    ::std::error_domains::__herbceptions_detail::__format_hex_value_max_size<
        T> +
    4u}; // '(' + "0x" + ')'

template <bool isebcdic, typename Chtype, typename T>
inline constexpr Chtype *__format_hex_value_full(Chtype *dest, T val) noexcept {
  using unsignedtype = ::std::make_unsigned_t<T>;
  using unsignedchtype = ::std::make_unsigned_t<Chtype>;
  static_assert(::std::is_integral_v<T>);
  if constexpr (!::std::is_signed_v<T>) {
    auto destend{dest + ::std::error_domains::__herbceptions_detail::
                            __format_hex_value_max_size_no_sign<T>};
    auto const last{destend};
    constexpr unsignedchtype chzero{isebcdic ? 0xF0 : u8'0'},
        chA{isebcdic ? static_cast<unsignedtype>(0xC1 - 10u)
                     : (static_cast<unsignedtype>(u8'A') - 10u)};
    for (; dest != destend;) {
      --destend;
      auto remainder{val & 0xF};
      if (9u < remainder) {
        *destend =
            static_cast<Chtype>(static_cast<unsignedchtype>(remainder + chA));
      } else {
        *destend = static_cast<Chtype>(
            static_cast<unsignedchtype>(remainder + chzero));
      }
      val >>= 4;
    }
    return last;
  } else {
    unsignedtype uval{static_cast<unsignedtype>(val)};
    if (val < 0) {
      if constexpr (isebcdic) {
        *dest = u8'\x60';
      } else {
        *dest = u8'-';
      }
      ++dest;
      constexpr unsignedtype zero{};
      uval = static_cast<unsignedtype>(zero - uval);
    }
    return __format_hex_value_full<isebcdic, Chtype, unsignedtype>(dest, uval);
  }
}

template <bool isebcdic, typename Chtype, typename T>
inline constexpr Chtype *__format_hex_value_full_with_bracket(Chtype *dest,
                                                              T val) noexcept {
  constexpr Chtype leftbracket{
      isebcdic ? static_cast<Chtype>(0x4D)  // EBCDIC '('
               : static_cast<Chtype>(u8'(') // ASCII/UTF‑8 '('
  };

  constexpr Chtype rightbracket{
      isebcdic ? static_cast<Chtype>(0x5D)  // EBCDIC ')'
               : static_cast<Chtype>(u8')') // ASCII/UTF‑8 ')'
  };
  constexpr Chtype chzero{isebcdic ? static_cast<Chtype>(0xF0)
                                   : static_cast<Chtype>(u8'0')};
  constexpr Chtype chx{isebcdic ? static_cast<Chtype>(0xA7)
                                : static_cast<Chtype>(u8'x')};

  *dest = leftbracket;
  ++dest;
  *dest = chzero;
  ++dest;
  *dest = chx;
  ++dest;

  dest = ::std::error_domains::__herbceptions_detail::__format_hex_value_full<
      isebcdic>(dest, val);

  *dest = rightbracket;
  ++dest;

  return dest;
}

template <typename T>
inline constexpr ::std::size_t __format_decimal_value_max_size_no_sign{
    static_cast<::std::size_t>(::std::numeric_limits<T>::digits10) + 1u};

template <typename T>
inline constexpr ::std::size_t __format_decimal_value_max_size_with_brackets{
    ::std::error_domains::__herbceptions_detail::
        __format_decimal_value_max_size_no_sign<T> +
    2u};

/*
Writes "(<decimal digits>)" in the requested charset (ASCII/UTF-8 family or
EBCDIC). T must be unsigned. Returns one past the last written character.
*/
template <bool isebcdic, typename Chtype, typename T>
inline constexpr Chtype *
__format_decimal_value_full_with_bracket(Chtype *dest, T val) noexcept {
  using unsignedchtype = ::std::make_unsigned_t<Chtype>;
  static_assert(::std::is_integral_v<T>);
  static_assert(!::std::is_signed_v<T>);
  constexpr Chtype leftbracket{
      isebcdic ? static_cast<Chtype>(0x4D)  // EBCDIC '('
               : static_cast<Chtype>(u8'(') // ASCII/UTF‑8 '('
  };
  constexpr Chtype rightbracket{
      isebcdic ? static_cast<Chtype>(0x5D)  // EBCDIC ')'
               : static_cast<Chtype>(u8')') // ASCII/UTF‑8 ')'
  };
  constexpr unsignedchtype chzero{isebcdic
                                      ? static_cast<unsignedchtype>(0xF0)
                                      : static_cast<unsignedchtype>(u8'0')};
  *dest = leftbracket;
  ++dest;
  ::std::size_t ndigits{1u};
  {
    T v{val};
    while (static_cast<T>(10u) <= v) {
      v /= static_cast<T>(10u);
      ++ndigits;
    }
  }
  Chtype *destend{dest + ndigits};
  {
    T v{val};
    for (Chtype *d{destend};;) {
      --d;
      *d = static_cast<Chtype>(static_cast<unsignedchtype>(
          chzero + static_cast<unsignedchtype>(v % static_cast<T>(10u))));
      v /= static_cast<T>(10u);
      if (v == static_cast<T>(0)) {
        break;
      }
    }
  }
  *destend = rightbracket;
  ++destend;
  return destend;
}

} // namespace std::error_domains::__herbceptions_detail
