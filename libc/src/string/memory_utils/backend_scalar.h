//===-- Elementary operations for native scalar types ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BACKEND_SCALAR_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BACKEND_SCALAR_H

#include "src/__support/CPP/TypeTraits.h" // ConditionalType, EnableIfType
#include "src/__support/endian.h"

namespace __llvm_libc {

struct Scalar64BitBackend {
  static constexpr bool IS_BACKEND_TYPE = true;

  template <typename T>
  static constexpr bool IsScalarType =
      cpp::IsSameV<T, uint8_t> || cpp::IsSameV<T, uint16_t> ||
      cpp::IsSameV<T, uint32_t> || cpp::IsSameV<T, uint64_t>;

  template <typename T, Temporality TS, Aligned AS>
  static inline T load(const T *src) {
    static_assert(IsScalarType<T>);
    static_assert(TS == Temporality::TEMPORAL,
                  "Scalar load does not support non-temporal access");
    return *src;
  }

  template <typename T, Temporality TS, Aligned AS>
  static inline void store(T *dst, T value) {
    static_assert(IsScalarType<T>);
    static_assert(TS == Temporality::TEMPORAL,
                  "Scalar store does not support non-temporal access");
    *dst = value;
  }

  template <typename T> static inline T splat(ubyte value) {
    static_assert(IsScalarType<T>);
    return (T(~0ULL) / T(0xFF)) * T(value);
  }

  template <typename T> static inline uint64_t notEquals(T v1, T v2) {
    static_assert(IsScalarType<T>);
    return v1 ^ v2;
  }

  template <typename T> static inline int32_t threeWayCmp(T v1, T v2) {
    DeferredStaticAssert("not implemented");
  }

  // Returns the type to use to consume Size bytes.
  template <size_t Size>
  using getNextType = cpp::ConditionalType<
      Size >= 8, uint64_t,
      cpp::ConditionalType<Size >= 4, uint32_t,
                           cpp::ConditionalType<Size >= 2, uint16_t, uint8_t>>>;
};

template <>
int32_t inline Scalar64BitBackend::threeWayCmp<uint8_t>(uint8_t a, uint8_t b) {
  const int16_t la = Endian::to_big_endian(a);
  const int16_t lb = Endian::to_big_endian(b);
  return la - lb;
}
template <>
int32_t inline Scalar64BitBackend::threeWayCmp<uint16_t>(uint16_t a,
                                                         uint16_t b) {
  const int32_t la = Endian::to_big_endian(a);
  const int32_t lb = Endian::to_big_endian(b);
  return la - lb;
}
template <>
int32_t inline Scalar64BitBackend::threeWayCmp<uint32_t>(uint32_t a,
                                                         uint32_t b) {
  const uint32_t la = Endian::to_big_endian(a);
  const uint32_t lb = Endian::to_big_endian(b);
  return la > lb ? 1 : la < lb ? -1 : 0;
}
template <>
int32_t inline Scalar64BitBackend::threeWayCmp<uint64_t>(uint64_t a,
                                                         uint64_t b) {
  const uint64_t la = Endian::to_big_endian(a);
  const uint64_t lb = Endian::to_big_endian(b);
  return la > lb ? 1 : la < lb ? -1 : 0;
}

namespace scalar {
using _1 = SizedOp<Scalar64BitBackend, 1>;
using _2 = SizedOp<Scalar64BitBackend, 2>;
using _3 = SizedOp<Scalar64BitBackend, 3>;
using _4 = SizedOp<Scalar64BitBackend, 4>;
using _8 = SizedOp<Scalar64BitBackend, 8>;
using _16 = SizedOp<Scalar64BitBackend, 16>;
using _32 = SizedOp<Scalar64BitBackend, 32>;
using _64 = SizedOp<Scalar64BitBackend, 64>;
using _128 = SizedOp<Scalar64BitBackend, 128>;
} // namespace scalar

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BACKEND_SCALAR_H
