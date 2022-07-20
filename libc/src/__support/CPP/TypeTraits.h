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

template <bool B, typename T> struct EnableIf;
template <typename T> struct EnableIf<true, T> {
  typedef T Type;
};

template <bool B, typename T>
using EnableIfType = typename EnableIf<B, T>::Type;

struct TrueValue {
  static constexpr bool Value = true;
};

struct FalseValue {
  static constexpr bool Value = false;
};

template <typename T> struct TypeIdentity {
  typedef T Type;
};

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
      IsSameV<unsigned long long, TypeNoCV> || IsSameV<bool, TypeNoCV> ||
      // We need to include UInt<128> and __uint128_t when available because
      // we want to unittest UInt<128>. If we include only UInt128, then on
      // platform where it resolves to __uint128_t, we cannot unittest
      // UInt<128>.
      IsSameV<__llvm_libc::cpp::UInt<128>, TypeNoCV>
#ifdef __SIZEOF_INT128__
      || IsSameV<__int128_t, TypeNoCV> || IsSameV<__uint128_t, TypeNoCV>
#endif
      ;
};

template <typename Type> struct IsEnum {
  static constexpr bool Value = __is_enum(Type);
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

template <typename Type> struct IsSigned {
  static constexpr bool Value =
      IsArithmetic<Type>::Value && (Type(-1) < Type(0));
  constexpr operator bool() const { return Value; }
  constexpr bool operator()() const { return Value; }
};

template <typename Type> struct MakeUnsigned;
template <> struct MakeUnsigned<char> {
  using Type = unsigned char;
};
template <> struct MakeUnsigned<signed char> {
  using Type = unsigned char;
};
template <> struct MakeUnsigned<short> {
  using Type = unsigned short;
};
template <> struct MakeUnsigned<int> {
  using Type = unsigned int;
};
template <> struct MakeUnsigned<long> {
  using Type = unsigned long;
};
template <> struct MakeUnsigned<long long> {
  using Type = unsigned long long;
};
template <> struct MakeUnsigned<unsigned char> {
  using Type = unsigned char;
};
template <> struct MakeUnsigned<unsigned short> {
  using Type = unsigned short;
};
template <> struct MakeUnsigned<unsigned int> {
  using Type = unsigned int;
};
template <> struct MakeUnsigned<unsigned long> {
  using Type = unsigned long;
};
template <> struct MakeUnsigned<unsigned long long> {
  using Type = unsigned long long;
};
#ifdef __SIZEOF_INT128__
template <> struct MakeUnsigned<__int128_t> {
  using Type = __uint128_t;
};
template <> struct MakeUnsigned<__uint128_t> {
  using Type = __uint128_t;
};
#endif

template <typename T> using MakeUnsignedType = typename MakeUnsigned<T>::Type;

// Compile time type selection.
template <bool _, class TrueT, class FalseT> struct Conditional {
  using type = TrueT;
};
template <class TrueT, class FalseT> struct Conditional<false, TrueT, FalseT> {
  using type = FalseT;
};
template <bool Cond, typename TrueT, typename FalseT>
using ConditionalType = typename Conditional<Cond, TrueT, FalseT>::type;

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_TYPETRAITS_H
