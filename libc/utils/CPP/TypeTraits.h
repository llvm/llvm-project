//===----------------- Self contained C++ type traits -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

template <typename Type> struct IsIntegral : public FalseValue {};
template <> struct IsIntegral<char> : public TrueValue {};
template <> struct IsIntegral<signed char> : public TrueValue {};
template <> struct IsIntegral<unsigned char> : public TrueValue {};
template <> struct IsIntegral<short> : public TrueValue {};
template <> struct IsIntegral<unsigned short> : public TrueValue {};
template <> struct IsIntegral<int> : public TrueValue {};
template <> struct IsIntegral<unsigned int> : public TrueValue {};
template <> struct IsIntegral<long> : public TrueValue {};
template <> struct IsIntegral<unsigned long> : public TrueValue {};
template <> struct IsIntegral<long long> : public TrueValue {};
template <> struct IsIntegral<unsigned long long> : public TrueValue {};
template <> struct IsIntegral<bool> : public TrueValue {};

template <typename T> struct IsPointerType : public FalseValue {};
template <typename T> struct IsPointerType<T *> : public TrueValue {};

template <typename T1, typename T2> struct IsSame : public FalseValue {};
template <typename T> struct IsSame<T, T> : public TrueValue {};

} // namespace cpp
} // namespace __llvm_libc
