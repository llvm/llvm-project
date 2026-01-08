//===-- 256-bit storage for StdRng - wctype conversion ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is a portable implementation of a 256-bit vector storage implemented
// with static arrays, parallel of <immintrin.h>'s AVX256 which works with
// `constexpr` code.
// Only little-endian is supported (runtime code is not affected by this).

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_VEC256_STORAGE_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_VEC256_STORAGE_H

#include "vec128_storage.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace random {

namespace vector_storage {

union vec256_storage {
  mutable cpp::array<uint32_t, 8> u32x8;

  LIBC_INLINE constexpr operator cpp::array<uint32_t, 8>() const {
    return this->u32x8;
  }

  LIBC_INLINE constexpr vec256_storage() : u32x8() {}
  LIBC_INLINE static constexpr vec256_storage
  construct_from_vec128(vec128_storage &&lo, vec128_storage &&hi) {
    vec256_storage r{{}};
    for (size_t i = 0; i < 4; i++) {
      r.u32x8[i] = lo.u32x4[i];
    }
    for (size_t i = 0; i < 4; i++) {
      r.u32x8[i + 4] = hi.u32x4[i];
    }
    return r;
  }

  LIBC_INLINE constexpr vec256_storage(cpp::array<uint32_t, 8> &&x)
      : u32x8(x) {}

  LIBC_INLINE constexpr vec256_storage shuffle_lane_words3012() const;
  LIBC_INLINE constexpr vec256_storage shuffle_lane_words2301() const;
  LIBC_INLINE constexpr vec256_storage shuffle_lane_words1230() const;
  LIBC_INLINE constexpr vec256_storage to_lanes() const;
};

} // namespace vector_storage

} // namespace random

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_VEC256_STORAGE_H
