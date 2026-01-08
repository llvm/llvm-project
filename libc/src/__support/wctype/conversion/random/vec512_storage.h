//===-- 512-bit storage for StdRng - wctype conversion ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is a portable implementation of a 512-bit vector storage implemented
// with static arrays, works with `constexpr` code.
// Only little-endian is supported (runtime code is not affected by this).

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_VEC512_STORAGE_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_VEC512_STORAGE_H

#include "vec256_storage.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace random {

namespace vector_storage {

union vec512_storage {
  mutable cpp::array<uint32_t, 16> u32x16;

  LIBC_INLINE static constexpr vec512_storage
  construct_from_vec256(const vec256_storage &lo, const vec256_storage &hi);

  LIBC_INLINE constexpr operator cpp::array<uint32_t, 16>() const {
    return this->u32x16;
  }

  LIBC_INLINE static constexpr vec512_storage
  new128(cpp::array<uint32_t, 16> &&xs) {
    return vec512_storage{xs};
  }

  LIBC_INLINE static constexpr vec512_storage new128(vec128_storage i,
                                                     vec128_storage j,
                                                     vec128_storage k,
                                                     vec128_storage l);

  LIBC_INLINE constexpr vec512_storage unpack() const { return *this; }

  LIBC_INLINE constexpr const vec512_storage &
  operator+=(vec512_storage &rhs) const;

  LIBC_INLINE constexpr vec512_storage operator+(const vec512_storage &) const;
  LIBC_INLINE constexpr vec512_storage operator^(vec512_storage &rhs) const;

  LIBC_INLINE constexpr vec512_storage rotate_each_word_right16() const;
  LIBC_INLINE constexpr vec512_storage rotate_each_word_right20() const;
  LIBC_INLINE constexpr vec512_storage rotate_each_word_right24() const;
  LIBC_INLINE constexpr vec512_storage rotate_each_word_right25() const;

  LIBC_INLINE constexpr vec512_storage shuffle_lane_words3012() const;
  LIBC_INLINE constexpr vec512_storage shuffle_lane_words2301() const;
  LIBC_INLINE constexpr vec512_storage shuffle_lane_words1230() const;

  LIBC_INLINE static constexpr cpp::array<vec512_storage, 4>
  transpose4(const vec512_storage &a, const vec512_storage &b,
             const vec512_storage &c, const vec512_storage &d);

  LIBC_INLINE constexpr conversion_utils::Slice<uint32_t> to_scalars() const {
    return conversion_utils::Slice<uint32_t>(this->u32x16.data(),
                                             this->u32x16.size());
  }

  LIBC_INLINE constexpr vec512_storage to_lanes() const { return *this; }
};

} // namespace vector_storage

} // namespace random

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_VEC512_STORAGE_H
