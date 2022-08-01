//===-- Self contained C++ type traits --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_TYPETRAITS_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_TYPETRAITS_H

#include "UInt.h"

namespace __llvm_libc {
namespace cpp {

template <bool B, typename T> struct enable_if;
template <typename T> struct enable_if<true, T> {
  using type = T;
};
template <bool B, typename T = void>
using enable_if_t = typename enable_if<B, T>::type;

template <typename T, T v> struct integral_constant {
  using value_type = T;
  static constexpr T value = v;
};
using true_type = cpp::integral_constant<bool, true>;
using false_type = cpp::integral_constant<bool, false>;

template <typename T> struct type_identity {
  using type = T;
};

template <typename T, typename U> struct is_same : cpp::false_type {};
template <typename T> struct is_same<T, T> : cpp::true_type {};
template <typename T, typename U>
inline constexpr bool is_same_v = is_same<T, U>::value;

template <typename T> struct remove_cv : public type_identity<T> {};
template <typename T> struct remove_cv<const T> : public type_identity<T> {};
template <typename T> struct remove_cv<volatile T> : public type_identity<T> {};
template <typename T>
struct remove_cv<const volatile T> : public type_identity<T> {};
template <typename T> using remove_cv_t = typename remove_cv<T>::type;

template <typename T> struct is_integral {
private:
  using unqualified_type = remove_cv_t<T>;

public:
  static constexpr bool value =
      is_same_v<char, unqualified_type> ||
      is_same_v<signed char, unqualified_type> ||
      is_same_v<unsigned char, unqualified_type> ||
      is_same_v<short, unqualified_type> ||
      is_same_v<unsigned short, unqualified_type> ||
      is_same_v<int, unqualified_type> ||
      is_same_v<unsigned int, unqualified_type> ||
      is_same_v<long, unqualified_type> ||
      is_same_v<unsigned long, unqualified_type> ||
      is_same_v<long long, unqualified_type> ||
      is_same_v<unsigned long long, unqualified_type> ||
      is_same_v<bool, unqualified_type> ||
      // We need to include UInt<128> and __uint128_t when available because
      // we want to unittest UInt<128>. If we include only UInt128, then on
      // platform where it resolves to __uint128_t, we cannot unittest
      // UInt<128>.
      is_same_v<__llvm_libc::cpp::UInt<128>, unqualified_type>
#ifdef __SIZEOF_INT128__
      || is_same_v<__int128_t, unqualified_type> ||
      is_same_v<__uint128_t, unqualified_type>
#endif
      ;
};
template <typename T>
inline constexpr bool is_integral_v = is_integral<T>::value;

template <typename T> struct is_enum {
  static constexpr bool value = __is_enum(T);
};

template <typename T> struct is_pointer : cpp::false_type {};
template <typename T> struct is_pointer<T *> : cpp::true_type {};
template <typename T> struct is_pointer<T *const> : cpp::true_type {};
template <typename T> struct is_pointer<T *volatile> : cpp::true_type {};
template <typename T> struct is_pointer<T *const volatile> : cpp::true_type {};
template <typename T> inline constexpr bool is_pointer_v = is_pointer<T>::value;

template <typename T> struct is_floating_point {
private:
  using unqualified_type = remove_cv_t<T>;

public:
  static constexpr bool value = is_same_v<float, unqualified_type> ||
                                is_same_v<double, unqualified_type> ||
                                is_same_v<long double, unqualified_type>;
};
template <typename T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

template <typename T> struct is_arithmetic {
  static constexpr bool value =
      is_integral<T>::value || is_floating_point<T>::value;
};
template <typename T>
inline constexpr bool is_arithmetic_v = is_arithmetic<T>::value;

template <typename T> struct is_signed {
  static constexpr bool value = is_arithmetic<T>::value && (T(-1) < T(0));
  constexpr operator bool() const { return value; }
  constexpr bool operator()() const { return value; }
};
template <typename T> inline constexpr bool is_signed_v = is_signed<T>::value;

template <typename T> struct make_unsigned;
template <> struct make_unsigned<char> {
  using type = unsigned char;
};
template <> struct make_unsigned<signed char> {
  using type = unsigned char;
};
template <> struct make_unsigned<short> {
  using type = unsigned short;
};
template <> struct make_unsigned<int> {
  using type = unsigned int;
};
template <> struct make_unsigned<long> {
  using type = unsigned long;
};
template <> struct make_unsigned<long long> {
  using type = unsigned long long;
};
template <> struct make_unsigned<unsigned char> {
  using type = unsigned char;
};
template <> struct make_unsigned<unsigned short> {
  using type = unsigned short;
};
template <> struct make_unsigned<unsigned int> {
  using type = unsigned int;
};
template <> struct make_unsigned<unsigned long> {
  using type = unsigned long;
};
template <> struct make_unsigned<unsigned long long> {
  using type = unsigned long long;
};
#ifdef __SIZEOF_INT128__
template <> struct make_unsigned<__int128_t> {
  using type = __uint128_t;
};
template <> struct make_unsigned<__uint128_t> {
  using type = __uint128_t;
};
#endif
template <typename T> using make_unsigned_t = typename make_unsigned<T>::type;

// Compile time type selection.
template <bool B, typename T, typename F> struct conditional {
  using type = T;
};
template <typename T, typename F> struct conditional<false, T, F> {
  using type = F;
};
template <bool B, typename T, typename F>
using conditional_t = typename conditional<B, T, F>::type;

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_TYPETRAITS_H
