//===-- A self contained equivalent of cstddef ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_BYTE_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_BYTE_H

#include "type_traits.h" // For enable_if_t, is_integral_v.

namespace __llvm_libc::cpp {

enum class byte : unsigned char {};

template <class IntegerType>
constexpr enable_if_t<is_integral_v<IntegerType>, byte>
operator>>(byte b, IntegerType shift) noexcept {
  return static_cast<byte>(static_cast<unsigned char>(b) >> shift);
}
template <class IntegerType>
constexpr enable_if_t<is_integral_v<IntegerType>, byte &>
operator>>=(byte &b, IntegerType shift) noexcept {
  return b = b >> shift;
}
template <class IntegerType>
constexpr enable_if_t<is_integral_v<IntegerType>, byte>
operator<<(byte b, IntegerType shift) noexcept {
  return static_cast<byte>(static_cast<unsigned char>(b) << shift);
}
template <class IntegerType>
constexpr enable_if_t<is_integral_v<IntegerType>, byte &>
operator<<=(byte &b, IntegerType shift) noexcept {
  return b = b << shift;
}
constexpr byte operator|(byte l, byte r) noexcept {
  return static_cast<byte>(static_cast<unsigned char>(l) |
                           static_cast<unsigned char>(r));
}
constexpr byte &operator|=(byte &l, byte r) noexcept { return l = l | r; }
constexpr byte operator&(byte l, byte r) noexcept {
  return static_cast<byte>(static_cast<unsigned char>(l) &
                           static_cast<unsigned char>(r));
}
constexpr byte &operator&=(byte &l, byte r) noexcept { return l = l & r; }
constexpr byte operator^(byte l, byte r) noexcept {
  return static_cast<byte>(static_cast<unsigned char>(l) ^
                           static_cast<unsigned char>(r));
}
constexpr byte &operator^=(byte &l, byte r) noexcept { return l = l ^ r; }
constexpr byte operator~(byte b) noexcept {
  return static_cast<byte>(~static_cast<unsigned char>(b));
}
template <typename IntegerType>
constexpr enable_if_t<is_integral_v<IntegerType>, IntegerType>
to_integer(byte b) noexcept {
  return static_cast<IntegerType>(b);
}

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_BYTE_H
