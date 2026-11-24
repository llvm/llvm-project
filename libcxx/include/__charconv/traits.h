// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHARCONV_TRAITS
#define _LIBCPP___CHARCONV_TRAITS

#include <__assert>
#include <__bit/countl.h>
#include <__charconv/tables.h>
#include <__charconv/to_chars_base_10.h>
#include <__config>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_unsigned.h>
#include <__utility/checked_int.h>
#include <cstdint>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __itoa {

template <typename _Tp, typename = void>
struct _LIBCPP_HIDDEN __traits_base;

template <typename _Tp>
struct _LIBCPP_HIDDEN __traits_base<_Tp, __enable_if_t<sizeof(_Tp) <= sizeof(uint32_t)> > {
  using type = uint32_t;

  /// The width estimation using a log10 algorithm.
  ///
  /// The algorithm is based on
  /// http://graphics.stanford.edu/~seander/bithacks.html#IntegerLog10
  /// Instead of using IntegerLogBase2 it uses __countl_zero.
  static _LIBCPP_CONSTEXPR_SINCE_CXX23 int __width(_Tp __v) {
    auto __t = (32 - std::__countl_zero(static_cast<type>(__v | 1))) * 1233 >> 12;
    return __t - (__v < __itoa::__pow10_32[__t]) + 1;
  }

  static _LIBCPP_CONSTEXPR_SINCE_CXX23 char* __convert(char* __p, _Tp __v) { return __itoa::__base_10_u32(__p, __v); }

  static _LIBCPP_CONSTEXPR_SINCE_CXX23 decltype(__pow10_32)& __pow() { return __itoa::__pow10_32; }
};

template <typename _Tp>
struct _LIBCPP_HIDDEN __traits_base<_Tp, __enable_if_t<sizeof(_Tp) == sizeof(uint64_t)> > {
  using type = uint64_t;

  /// The width estimation using a log10 algorithm.
  ///
  /// The algorithm is based on
  /// http://graphics.stanford.edu/~seander/bithacks.html#IntegerLog10
  /// Instead of using IntegerLogBase2 it uses __countl_zero.
  static _LIBCPP_CONSTEXPR_SINCE_CXX23 int __width(_Tp __v) {
    auto __t = (64 - std::__countl_zero(static_cast<type>(__v | 1))) * 1233 >> 12;
    return __t - (__v < __itoa::__pow10_64[__t]) + 1;
  }

  static _LIBCPP_CONSTEXPR_SINCE_CXX23 char* __convert(char* __p, _Tp __v) { return __itoa::__base_10_u64(__p, __v); }

  static _LIBCPP_CONSTEXPR_SINCE_CXX23 decltype(__pow10_64)& __pow() { return __itoa::__pow10_64; }
};

#if _LIBCPP_HAS_INT128
template <typename _Tp>
struct _LIBCPP_HIDDEN __traits_base<_Tp, __enable_if_t<sizeof(_Tp) == sizeof(__uint128_t)> > {
  using type = __uint128_t;

  /// The width estimation using a log10 algorithm.
  ///
  /// The algorithm is based on
  /// http://graphics.stanford.edu/~seander/bithacks.html#IntegerLog10
  /// Instead of using IntegerLogBase2 it uses __countl_zero.
  static _LIBCPP_CONSTEXPR_SINCE_CXX23 int __width(_Tp __v) {
    _LIBCPP_ASSERT_INTERNAL(
        __v > numeric_limits<uint64_t>::max(), "The optimizations for this algorithm fail when this isn't true.");
    // There's always a bit set in the upper 64-bits.
    auto __t = (128 - std::__countl_zero(static_cast<uint64_t>(__v >> 64))) * 1233 >> 12;
    _LIBCPP_ASSERT_INTERNAL(__t >= __itoa::__pow10_128_offset, "Index out of bounds");
    // __t is adjusted since the lookup table misses the lower entries.
    return __t - (__v < __itoa::__pow10_128[__t - __itoa::__pow10_128_offset]) + 1;
  }

  static _LIBCPP_CONSTEXPR_SINCE_CXX23 char* __convert(char* __p, _Tp __v) { return __itoa::__base_10_u128(__p, __v); }

  // TODO FMT This pow function should get an index.
  // By moving this to its own header it can be reused by the pow function in to_chars_base_10.
  static _LIBCPP_CONSTEXPR_SINCE_CXX23 decltype(__pow10_128)& __pow() { return __itoa::__pow10_128; }
};
#endif

template <typename _Tp>
struct _LIBCPP_HIDDEN __traits : __traits_base<_Tp> {
  static _LIBCPP_CONSTEXPR const int digits = numeric_limits<_Tp>::digits10 + 1;
  using __traits_base<_Tp>::__pow;
  using typename __traits_base<_Tp>::type;

  // precondition: at least one non-zero character available
  //
  // Reads at most `digits` digits. On return __a + __b is the value of the digits read, where __b is the
  // last digit when `digits` digits were read (the caller checks __a + __b for overflow), and 0 otherwise.
  static _LIBCPP_CONSTEXPR_SINCE_CXX23 char const* __read(char const* __p, char const* __ep, type& __a, type& __b) {
    type __x = 0;
    int __n  = 0;
#if defined(_LIBCPP_COMPILER_CLANG_BASED)
    // Clang fully unrolls this loop (digits steps) by default, which makes 1-2 digit inputs slower; unroll by two.
#  pragma clang loop unroll_count(2)
#endif
    do {
      unsigned __c = static_cast<unsigned char>(*__p) - '0';
      if (__c > 9)
        break;
      if (__n == digits - 1) {
        // On overflow make __a + __b overflow too, so the caller reports out of range. The returned
        // pointer does not depend on the multiply: callers that chain parses through it would wait on
        // the whole digit chain otherwise.
        auto __res = __checked_int<type>(__x) * 10u;
        __a        = __res.__overflowed() ? numeric_limits<type>::max() : __res.__value();
        __b        = __res.__overflowed() ? numeric_limits<type>::max() : __c;
        return __p + 1;
      }
      __x = __x * 10 + __c;
      ++__p;
      ++__n;
    } while (__p != __ep);
    __a = __x;
    __b = 0;
    return __p;
  }
};

} // namespace __itoa

template <typename _Tp>
inline _LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_DISABLE_UBSAN_UNSIGNED_INTEGER_CHECK _Tp __complement(_Tp __x) {
  static_assert(is_unsigned<_Tp>::value, "cast to unsigned first");
  return _Tp(~__x + 1);
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CHARCONV_TRAITS
