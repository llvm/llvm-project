//===-- Self contained C++ type traits --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_TYPETRAITS_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_TYPETRAITS_H

namespace __llvm_libc {
namespace cpp {

template <typename T> struct type_identity {
  using type = T;
};

template <bool B, typename T> struct enable_if;
template <typename T> struct enable_if<true, T> : type_identity<T> {};
template <bool B, typename T = void>
using enable_if_t = typename enable_if<B, T>::type;

template <typename T, T v> struct integral_constant {
  using value_type = T;
  static constexpr T value = v;
};
using true_type = cpp::integral_constant<bool, true>;
using false_type = cpp::integral_constant<bool, false>;

template <typename T, typename U> struct is_same : cpp::false_type {};
template <typename T> struct is_same<T, T> : cpp::true_type {};
template <typename T, typename U>
inline constexpr bool is_same_v = is_same<T, U>::value;

template <class T> struct is_const : cpp::false_type {};
template <class T> struct is_const<const T> : cpp::true_type {};
template <class T> inline constexpr bool is_const_v = is_const<T>::value;

template <typename T> struct remove_cv : type_identity<T> {};
template <typename T> struct remove_cv<const T> : type_identity<T> {};
template <typename T> struct remove_cv<volatile T> : type_identity<T> {};
template <typename T> struct remove_cv<const volatile T> : type_identity<T> {};
template <typename T> using remove_cv_t = typename remove_cv<T>::type;

namespace details {
template <typename T, typename... Args> constexpr bool is_unqualified_any_of() {
  return (... || is_same_v<remove_cv_t<T>, Args>);
}
} // namespace details

template <typename T> struct is_integral {
  static constexpr bool value = details::is_unqualified_any_of<
      T,
#ifdef __SIZEOF_INT128__
      __int128_t, __uint128_t,
#endif
      char, signed char, unsigned char, short, unsigned short, int,
      unsigned int, long, unsigned long, long long, unsigned long long, bool>();
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
  static constexpr bool value =
      details::is_unqualified_any_of<T, float, double, long double>();
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

template <typename T> struct is_unsigned {
  static constexpr bool value = is_arithmetic<T>::value && (T(-1) > T(0));
  constexpr operator bool() const { return value; }
  constexpr bool operator()() const { return value; }
};
template <typename T>
inline constexpr bool is_unsigned_v = is_unsigned<T>::value;

template <typename T> struct make_unsigned;
template <> struct make_unsigned<char> : type_identity<unsigned char> {};
template <> struct make_unsigned<signed char> : type_identity<unsigned char> {};
template <> struct make_unsigned<short> : type_identity<unsigned short> {};
template <> struct make_unsigned<int> : type_identity<unsigned int> {};
template <> struct make_unsigned<long> : type_identity<unsigned long> {};
template <>
struct make_unsigned<long long> : type_identity<unsigned long long> {};
template <>
struct make_unsigned<unsigned char> : type_identity<unsigned char> {};
template <>
struct make_unsigned<unsigned short> : type_identity<unsigned short> {};
template <> struct make_unsigned<unsigned int> : type_identity<unsigned int> {};
template <>
struct make_unsigned<unsigned long> : type_identity<unsigned long> {};
template <>
struct make_unsigned<unsigned long long> : type_identity<unsigned long long> {};
#ifdef __SIZEOF_INT128__
template <> struct make_unsigned<__int128_t> : type_identity<__uint128_t> {};
template <> struct make_unsigned<__uint128_t> : type_identity<__uint128_t> {};
#endif
template <typename T> using make_unsigned_t = typename make_unsigned<T>::type;

// Compile time type selection.
template <bool B, typename T, typename F>
struct conditional : type_identity<T> {};
template <typename T, typename F>
struct conditional<false, T, F> : type_identity<F> {};
template <bool B, typename T, typename F>
using conditional_t = typename conditional<B, T, F>::type;

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_TYPETRAITS_H
