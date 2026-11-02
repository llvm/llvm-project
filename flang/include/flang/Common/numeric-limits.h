//===-- include/flang/Common/numeric-limits.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_NUMERIC_LIMITS_H_
#define FORTRAN_COMMON_NUMERIC_LIMITS_H_

#include <limits>
#include <type_traits>

namespace Fortran::common {

namespace detail {

template <typename T> class numeric_limits_impl {
public:
  static constexpr bool is_specialized{false};
};

template <typename T> class make_unsigned_impl;

#if defined(__SIZEOF_INT128__)
// Handle discrepancy of support of 128-bit integers by compiler and
// standard library. The compiler may treat __int128 as a builtin type, but the
// standard library does not define std::numeric_limits for it. Two cases are
// known:
//
// 1. clang-cl supports __int128, but MSVC, and therefore its STL used by
//    clang-cl, does not.
//
// 2. Some versions of libstdc++ in strict mode (-std=c++NN)
//    intentionally remove any use of __int128, even though gcc does not make
//    such a distinction.
//
// Using __int128_t/__uint128_t typedefs; spelling out the __int128
// keyword is a warning "ISO C++ does not support ‘__int128’ for ‘type name’"
// under -Wpedantic

template <> class numeric_limits_impl<__uint128_t> {
public:
  using T = __uint128_t;

  static constexpr bool is_specialized{true};
  static constexpr bool is_signed{false};
  static constexpr bool is_integer{true};

  static constexpr T min() { return static_cast<T>(0); }
  static constexpr T max() { return ~static_cast<T>(0); }
  static constexpr T lowest() { return min(); }
};

template <> class numeric_limits_impl<__int128_t> {
public:
  using T = __int128_t;

  static constexpr bool is_specialized{true};
  static constexpr bool is_signed{true};
  static constexpr bool is_integer{true};

  static constexpr T min() {
    return static_cast<T>(static_cast<__uint128_t>(1) << 127u);
  }
  static constexpr T max() {
    return static_cast<T>(~(static_cast<__uint128_t>(1) << 127u));
  }
  static constexpr T lowest() { return min(); }
};

template <> class make_unsigned_impl<__int128_t> {
public:
  using type = __uint128_t;
};
template <> class make_unsigned_impl<__uint128_t> {
public:
  using type = __uint128_t;
};

#endif // defined(__SIZEOF_INT128__)

template <typename T>
using numeric_limits = numeric_limits_impl<std::remove_cv_t<T>>;

// Like std::make_unsigned: preserves the cv-qualifiers of T.
template <typename T> class make_unsigned {
  using Base = typename make_unsigned_impl<std::remove_cv_t<T>>::type;
  using C =
      std::conditional_t<std::is_const_v<T>, std::add_const_t<Base>, Base>;

public:
  using type =
      std::conditional_t<std::is_volatile_v<T>, std::add_volatile_t<C>, C>;
};

// True if make_unsigned_impl has been defined for T (ignoring cv-qualifiers).
template <typename T, typename = void>
inline constexpr bool has_make_unsigned_impl{false};
template <typename T>
inline constexpr bool has_make_unsigned_impl<T,
    std::void_t<typename make_unsigned_impl<std::remove_cv_t<T>>::type>>{true};

} // namespace detail

/// Same as std::numeric_limits, but also defined for 128-bit integers. While
/// std::numeric_limits is allowed to be extended for user-defined types such
/// as UnsignedInt128/SignedInt128 (C++ [namespace.std]), it is not for
/// non-standard compiler extensions such as
/// __int128/__uint128/__float128.
///
/// FIXME: __float128 not yet handled by detail::numeric_limits
///
/// Only defining members that are actually used in Flang/Flang-RT. Feel free to
/// add more members as needed.
template <typename T>
using numeric_limits =
    typename std::conditional_t<detail::numeric_limits<T>::is_specialized &&
            !std::numeric_limits<T>::is_specialized,
        detail::numeric_limits<T>, std::numeric_limits<T>>;

/// Same as the std:: versions, but guaranteed to be defined for 128-bit
/// integers (__int128/__uint128) as well. Uses the equivalent definitions of
/// detail::numeric_limits.
template <typename T>
inline constexpr bool is_arithmetic_v{numeric_limits<T>::is_specialized};
template <typename T>
inline constexpr bool is_integral_v{
    numeric_limits<T>::is_specialized && numeric_limits<T>::is_integer};
template <typename T>
inline constexpr bool is_signed_v{
    numeric_limits<T>::is_specialized && numeric_limits<T>::is_signed};
template <typename T>
inline constexpr bool is_unsigned_v{
    numeric_limits<T>::is_specialized && !numeric_limits<T>::is_signed};

/// Same as std::make_unsigned, but also defined for 128-bit integers. Prefers
/// detail::make_unsigned whenever it is defined for T, since std::make_unsigned
/// may lack __int128 support even if std::numeric_limits has it.
template <typename T>
using make_unsigned =
    typename std::conditional_t<detail::has_make_unsigned_impl<T>,
        detail::make_unsigned<T>, std::make_unsigned<T>>;

template <typename T> using make_unsigned_t = typename make_unsigned<T>::type;

} // namespace Fortran::common
#endif // FORTRAN_COMMON_NUMERIC_LIMITS_H_
