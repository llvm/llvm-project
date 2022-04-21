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

template <bool B, typename T> struct EnableIf;
template <typename T> struct EnableIf<true, T> { typedef T Type; };

template <bool B, typename T>
using EnableIfType = typename EnableIf<B, T>::Type;

struct TrueValue {
  static constexpr bool Value = true;
};

struct FalseValue {
  static constexpr bool Value = false;
};

template <typename T> struct TypeIdentity { typedef T Type; };

template <typename T1, typename T2> struct IsSame : public FalseValue {};
template <typename T> struct IsSame<T, T> : public TrueValue {};
template <typename T1, typename T2>
static constexpr bool IsSameV = IsSame<T1, T2>::Value;

template <typename T> struct RemoveCV : public TypeIdentity<T> {};
template <typename T> struct RemoveCV<const T> : public TypeIdentity<T> {};
template <typename T> struct RemoveCV<volatile T> : public TypeIdentity<T> {};
template <typename T>
struct RemoveCV<const volatile T> : public TypeIdentity<T> {};

template <typename T> using RemoveCVType = typename RemoveCV<T>::Type;

template <typename Type> struct IsIntegral {
  using TypeNoCV = RemoveCVType<Type>;
  static constexpr bool Value =
      IsSameV<char, TypeNoCV> || IsSameV<signed char, TypeNoCV> ||
      IsSameV<unsigned char, TypeNoCV> || IsSameV<short, TypeNoCV> ||
      IsSameV<unsigned short, TypeNoCV> || IsSameV<int, TypeNoCV> ||
      IsSameV<unsigned int, TypeNoCV> || IsSameV<long, TypeNoCV> ||
      IsSameV<unsigned long, TypeNoCV> || IsSameV<long long, TypeNoCV> ||
      IsSameV<unsigned long long, TypeNoCV> || IsSameV<bool, TypeNoCV>
#ifdef __SIZEOF_INT128__
      || IsSameV<__uint128_t, TypeNoCV> || IsSameV<__int128_t, TypeNoCV>
#endif
      ;
};

template <typename T> struct IsPointerTypeNoCV : public FalseValue {};
template <typename T> struct IsPointerTypeNoCV<T *> : public TrueValue {};
template <typename T> struct IsPointerType {
  static constexpr bool Value = IsPointerTypeNoCV<RemoveCVType<T>>::Value;
};

template <typename Type> struct IsFloatingPointType {
  using TypeNoCV = RemoveCVType<Type>;
  static constexpr bool Value = IsSame<float, TypeNoCV>::Value ||
                                IsSame<double, TypeNoCV>::Value ||
                                IsSame<long double, TypeNoCV>::Value;
};

template <typename Type> struct IsArithmetic {
  static constexpr bool Value =
      IsIntegral<Type>::Value || IsFloatingPointType<Type>::Value;
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_TYPETRAITS_H
