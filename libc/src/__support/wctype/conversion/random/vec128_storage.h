//===-- 128-bit storage for StdRng - wctype conversion ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is a portable implementation of a 128-bit vector storage implemented
// with static arrays, parallel of <immintrin.h>'s `__m128i` which works with
// `constexpr` code.
// Only little-endian is supported (runtime code is not affected by this).

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_VEC128_STORAGE_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_VEC128_STORAGE_H

#include "src/__support/CPP/array.h"
#include "src/__support/wctype/conversion/utils/slice.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace random {

namespace vector_storage {

union vec128_storage {
  mutable cpp::array<uint32_t, 4> u32x4;

  LIBC_INLINE constexpr vec128_storage(cpp::array<uint32_t, 4> &&x)
      : u32x4(x) {}
  LIBC_INLINE constexpr vec128_storage(cpp::array<uint32_t, 4> &x) : u32x4(x) {}
  LIBC_INLINE constexpr vec128_storage() : u32x4() {}

  LIBC_INLINE constexpr operator cpp::array<uint32_t, 4>() const {
    return this->u32x4;
  }

  LIBC_INLINE constexpr cpp::array<uint32_t, 4> to_lanes() const {
    return this->u32x4;
  }

  LIBC_INLINE static constexpr vec128_storage
  from_lanes(cpp::array<uint32_t, 4> &&xs) {
    return vec128_storage(xs);
  }

  LIBC_INLINE static constexpr auto from_lanes(cpp::array<uint64_t, 2> &&xs) {
    cpp::array<uint32_t, 4> x = {
        static_cast<uint32_t>(xs[0]), static_cast<uint32_t>(xs[0] >> 32),
        static_cast<uint32_t>(xs[1]), static_cast<uint32_t>(xs[1] >> 32)};
    return vec128_storage(x);
  }

  LIBC_INLINE static constexpr auto
  read_le(conversion_utils::Slice<uint8_t> x) {
    LIBC_ASSERT(x.size() == 16);
    vec128_storage v = cpp::array<uint32_t, 4>{0};
    uint32_t *dst = v.u32x4.data();
    uint8_t *src = x.data();
    for (uint8_t i = 0; i < 4; ++i)
      dst[i] = src[i * 4] | (src[i * 4 + 1] << 8) | (src[i * 4 + 2] << 16) |
               (src[i * 4 + 3] << 24);
    return v;
  }
};

} // namespace vector_storage

} // namespace random

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_VEC128_STORAGE_H
