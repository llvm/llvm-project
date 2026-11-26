//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_UINT_FAST_BITS_H
#define _LIBCPP___TYPE_TRAITS_UINT_FAST_BITS_H

#include <__config>
#include <__cstddef/size_t.h>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 20

_LIBCPP_BEGIN_NAMESPACE_STD

#  ifdef __BITINT_MAXWIDTH__

template <size_t _Bits>
consteval auto __bit_uint_unpadded() {
  if constexpr (_Bits <= __BITINT_MAXWIDTH__) {
    constexpr size_t __storage_bits = sizeof(unsigned _BitInt(_Bits)) * __CHAR_BIT__;
    if constexpr (__storage_bits <= __BITINT_MAXWIDTH__)
      return static_cast<unsigned _BitInt(__storage_bits)>(0);
    else
      return void();
  } else
    return void();
}

/// Alias template that yields an unsigned _BitInt whose width is at least
/// \p _Bits and which has no padding bits. For example,
/// __bit_uint_unpadded_t<65> is typically unsigned _BitInt(96) or unsigned
/// _BitInt(128), depending on whether the architecture is 32-bit or 64-bit,
/// respectively. If no such type exists, the alias template denotes void.
template <size_t _Bits>
using __bit_uint_unpadded_t = decltype(__bit_uint_unpadded<_Bits>());

#  endif // defined(__BITINT_MAXWIDTH__)

template <size_t _Bits>
consteval auto __uint_fast_bits() {
  if constexpr (_Bits <= 8)
    return uint_fast8_t{};
  else if constexpr (_Bits <= 16)
    return uint_fast16_t{};
  else if constexpr (_Bits <= 32)
    return uint_fast32_t{};
  else if constexpr (_Bits <= 64)
    return uint_fast64_t{};
#  if _LIBCPP_HAS_INT128
  else if constexpr (_Bits <= 128)
    return static_cast<unsigned __int128>(0);
#  endif
#  ifdef __BITINT_MAXWIDTH__
  else
    return __bit_uint_unpadded<_Bits>();
#  else
  else
    return void();
#  endif
};

/// Alias template that yields a "fast" unsigned integer with at least the
/// specified \p _Bits width. If one of the uint_fastN_t aliases is sufficiently
/// wide, that alias is used. Otherwise, unsigned __int128 and unsigned _BitInt
/// types are used (if available). If no such type exists, the alias template
/// denotes void.
template <size_t _Bits>
using __uint_fast_bits_t = decltype(__uint_fast_bits<_Bits>());

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

#endif // _LIBCPP___TYPE_TRAITS_UINT_FAST_BITS_H
