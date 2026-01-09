//===-- 256-bit storage implementation --------------------------*- C++ -*-===//
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

#include "vec256_storage.h"
#include "imm.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace random {

namespace vector_storage {

LIBC_INLINE constexpr vec256_storage
vec256_storage::shuffle_lane_words3012() const {
  return immintrin::mm256_shuffle_epi32(*this, 0b0011'1001);
}

LIBC_INLINE constexpr vec256_storage
vec256_storage::shuffle_lane_words2301() const {
  return immintrin::mm256_shuffle_epi32(*this, 0b0100'1110);
}

LIBC_INLINE constexpr vec256_storage
vec256_storage::shuffle_lane_words1230() const {
  return immintrin::mm256_shuffle_epi32(*this, 0b1001'0011);
}

LIBC_INLINE constexpr vec256_storage vec256_storage::to_lanes() const {
  auto lo = immintrin::mm256_extracti128_si256(*this, 0);
  auto hi = immintrin::mm256_extracti128_si256(*this, 1);
  return vec256_storage{{
      lo.u32x4[0],
      lo.u32x4[1],
      lo.u32x4[2],
      lo.u32x4[3],
      hi.u32x4[0],
      hi.u32x4[1],
      hi.u32x4[2],
      hi.u32x4[3],
  }};
}

} // namespace vector_storage

} // namespace random

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL
