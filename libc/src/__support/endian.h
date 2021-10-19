//===-- Endianness support ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_ENDIAN_H
#define LLVM_LIBC_SRC_SUPPORT_ENDIAN_H

#include <stdint.h>

namespace __llvm_libc {

// We rely on compiler preprocessor defines to allow for cross compilation.
#if !defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__) ||           \
    !defined(__ORDER_BIG_ENDIAN__)
#error "Missing preprocessor definitions for endianness detection."
#endif

namespace internal {

// Converts uint8_t, uint16_t, uint32_t, uint64_t to its big or little endian
// counterpart.
// We use explicit template specialization:
// - to prevent accidental integer promotion.
// - to prevent fallback in (unlikely) case of middle-endianness.

template <unsigned ORDER> struct Endian {
  static constexpr const bool isLittle = ORDER == __ORDER_LITTLE_ENDIAN__;
  static constexpr const bool isBig = ORDER == __ORDER_BIG_ENDIAN__;
  template <typename T> static T ToBigEndian(T value);
  template <typename T> static T ToLittleEndian(T value);
};

// Little Endian specializations
template <>
template <>
inline uint8_t
Endian<__ORDER_LITTLE_ENDIAN__>::ToBigEndian<uint8_t>(uint8_t v) {
  return v;
}
template <>
template <>
inline uint8_t
Endian<__ORDER_LITTLE_ENDIAN__>::ToLittleEndian<uint8_t>(uint8_t v) {
  return v;
}
template <>
template <>
inline uint16_t
Endian<__ORDER_LITTLE_ENDIAN__>::ToBigEndian<uint16_t>(uint16_t v) {
  return __builtin_bswap16(v);
}
template <>
template <>
inline uint16_t
Endian<__ORDER_LITTLE_ENDIAN__>::ToLittleEndian<uint16_t>(uint16_t v) {
  return v;
}
template <>
template <>
inline uint32_t
Endian<__ORDER_LITTLE_ENDIAN__>::ToBigEndian<uint32_t>(uint32_t v) {
  return __builtin_bswap32(v);
}
template <>
template <>
inline uint32_t
Endian<__ORDER_LITTLE_ENDIAN__>::ToLittleEndian<uint32_t>(uint32_t v) {
  return v;
}
template <>
template <>
inline uint64_t
Endian<__ORDER_LITTLE_ENDIAN__>::ToBigEndian<uint64_t>(uint64_t v) {
  return __builtin_bswap64(v);
}
template <>
template <>
inline uint64_t
Endian<__ORDER_LITTLE_ENDIAN__>::ToLittleEndian<uint64_t>(uint64_t v) {
  return v;
}

// Big Endian specializations
template <>
template <>
inline uint8_t Endian<__ORDER_BIG_ENDIAN__>::ToBigEndian<uint8_t>(uint8_t v) {
  return v;
}
template <>
template <>
inline uint8_t
Endian<__ORDER_BIG_ENDIAN__>::ToLittleEndian<uint8_t>(uint8_t v) {
  return v;
}
template <>
template <>
inline uint16_t
Endian<__ORDER_BIG_ENDIAN__>::ToBigEndian<uint16_t>(uint16_t v) {
  return v;
}
template <>
template <>
inline uint16_t
Endian<__ORDER_BIG_ENDIAN__>::ToLittleEndian<uint16_t>(uint16_t v) {
  return __builtin_bswap16(v);
}
template <>
template <>
inline uint32_t
Endian<__ORDER_BIG_ENDIAN__>::ToBigEndian<uint32_t>(uint32_t v) {
  return v;
}
template <>
template <>
inline uint32_t
Endian<__ORDER_BIG_ENDIAN__>::ToLittleEndian<uint32_t>(uint32_t v) {
  return __builtin_bswap32(v);
}
template <>
template <>
inline uint64_t
Endian<__ORDER_BIG_ENDIAN__>::ToBigEndian<uint64_t>(uint64_t v) {
  return v;
}
template <>
template <>
inline uint64_t
Endian<__ORDER_BIG_ENDIAN__>::ToLittleEndian<uint64_t>(uint64_t v) {
  return __builtin_bswap64(v);
}

} // namespace internal

using Endian = internal::Endian<__BYTE_ORDER__>;

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_ENDIAN_H
