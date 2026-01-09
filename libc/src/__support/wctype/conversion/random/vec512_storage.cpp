//===-- 512-bit storage implementation --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vec512_storage.h"
#include "imm.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace random {

namespace vector_storage {

LIBC_INLINE constexpr vec512_storage
vec512_storage::construct_from_vec256(const vec256_storage &lo,
                                      const vec256_storage &hi) {
  vec512_storage r{{}};
  for (size_t i = 0; i < 8; i++) {
    r.u32x16[i] = lo.u32x8[i];
  }
  for (size_t i = 0; i < 8; i++) {
    r.u32x16[8 + i] = hi.u32x8[i];
  }
  return r;
}

LIBC_INLINE constexpr vec512_storage vec512_storage::new128(vec128_storage i,
                                                            vec128_storage j,
                                                            vec128_storage k,
                                                            vec128_storage l) {
  vec512_storage r{{}};
  for (size_t a = 0; a < 4; a++) {
    r.u32x16[a] = i.u32x4[a];
    r.u32x16[a + 4] = j.u32x4[a];
    r.u32x16[a + 8] = k.u32x4[a];
    r.u32x16[a + 12] = l.u32x4[a];
  }
  return r;
}

LIBC_INLINE constexpr const vec512_storage &
vec512_storage::operator+=(vec512_storage &rhs) const {
  this->u32x16 = immintrin::mm256_add_epi32(*this, rhs).u32x16;
  return *this;
}

LIBC_INLINE constexpr vec512_storage
vec512_storage::operator+(const vec512_storage &rhs) const {
  return immintrin::mm256_add_epi32(*this, rhs);
}

LIBC_INLINE constexpr vec512_storage
vec512_storage::operator^(vec512_storage &rhs) const {
  return immintrin::mm256_xor_si256(*this, rhs);
}

LIBC_INLINE constexpr vec512_storage
vec512_storage::rotate_each_word_right16() const {
  auto constexpr K0 = 0x0d0c'0f0e'0908'0b0a;
  auto constexpr K1 = 0x0504'0706'0100'0302;

  vec256_storage lo{{}};
  vec256_storage hi{{}};

  for (size_t i = 0; i < 8; i++) {
    lo.u32x8[i] = this->u32x16[i];
  }
  for (size_t i = 0; i < 8; i++) {
    hi.u32x8[i] = this->u32x16[8 + i];
  }
  lo = immintrin::mm256_shuffle_epi8(
      lo, immintrin::mm256_set_epi64x(K0, K1, K0, K1));
  hi = immintrin::mm256_shuffle_epi8(
      hi, immintrin::mm256_set_epi64x(K0, K1, K0, K1));

  vec512_storage ret{{}};
  for (size_t i = 0; i < 8; i++) {
    ret.u32x16[i] = lo.u32x8[i];
  }

  for (size_t i = 0; i < 8; i++) {
    ret.u32x16[8 + i] = hi.u32x8[i];
  }

  return ret;
}

LIBC_INLINE constexpr vec512_storage
vec512_storage::rotate_each_word_right20() const {
  constexpr int32_t I = 20;

  vec256_storage lo{{}};
  vec256_storage hi{{}};

  for (size_t i = 0; i < 8; i++) {
    lo.u32x8[i] = this->u32x16[i];
  }
  for (size_t i = 0; i < 8; i++) {
    hi.u32x8[i] = this->u32x16[8 + i];
  }

  lo = immintrin::mm256_or_si256(immintrin::mm256_srli_epi32(lo, I),
                                 immintrin::mm256_slli_epi32(lo, 32 - I));
  hi = immintrin::mm256_or_si256(immintrin::mm256_srli_epi32(hi, I),
                                 immintrin::mm256_slli_epi32(hi, 32 - I));

  vec512_storage r{{}};

  for (size_t i = 0; i < 8; i++) {
    r.u32x16[i] = lo.u32x8[i];
  }
  for (size_t i = 0; i < 8; i++) {
    r.u32x16[8 + i] = hi.u32x8[i];
  }

  return r;
}

LIBC_INLINE constexpr vec512_storage
vec512_storage::rotate_each_word_right24() const {
  auto constexpr K0 = 0x0e0d'0c0f'0a09'080b;
  auto constexpr K1 = 0x0605'0407'0201'0003;

  vec256_storage lo{{}};
  vec256_storage hi{{}};

  for (size_t i = 0; i < 8; i++) {
    lo.u32x8[i] = this->u32x16[i];
  }
  for (size_t i = 0; i < 8; i++) {
    hi.u32x8[i] = this->u32x16[8 + i];
  }

  lo = immintrin::mm256_shuffle_epi8(
      lo, immintrin::mm256_set_epi64x(K0, K1, K0, K1));
  hi = immintrin::mm256_shuffle_epi8(
      hi, immintrin::mm256_set_epi64x(K0, K1, K0, K1));

  vec512_storage r{{}};

  for (size_t i = 0; i < 8; i++) {
    r.u32x16[i] = lo.u32x8[i];
  }
  for (size_t i = 0; i < 8; i++) {
    r.u32x16[8 + i] = hi.u32x8[i];
  }

  return r;
}

LIBC_INLINE constexpr vec512_storage
vec512_storage::rotate_each_word_right25() const {
  constexpr int32_t I = 25;
  vec256_storage lo{{}};
  vec256_storage hi{{}};

  for (size_t i = 0; i < 8; i++) {
    lo.u32x8[i] = this->u32x16[i];
  }
  for (size_t i = 0; i < 8; i++) {
    hi.u32x8[i] = this->u32x16[8 + i];
  }

  lo = immintrin::mm256_or_si256(immintrin::mm256_srli_epi32(lo, I),
                                 immintrin::mm256_slli_epi32(lo, 32 - I));
  hi = immintrin::mm256_or_si256(immintrin::mm256_srli_epi32(hi, I),
                                 immintrin::mm256_slli_epi32(hi, 32 - I));

  vec512_storage r{{}};

  for (size_t i = 0; i < 8; i++) {
    r.u32x16[i] = lo.u32x8[i];
  }
  for (size_t i = 0; i < 8; i++) {
    r.u32x16[8 + i] = hi.u32x8[i];
  }

  return r;
}

LIBC_INLINE constexpr vec512_storage
vec512_storage::shuffle_lane_words3012() const {
  vec256_storage lo{{}};
  vec256_storage hi{{}};

  for (size_t i = 0; i < 8; i++) {
    lo.u32x8[i] = this->u32x16[i];
  }
  for (size_t i = 0; i < 8; i++) {
    hi.u32x8[i] = this->u32x16[8 + i];
  }
  lo = lo.shuffle_lane_words3012();
  hi = hi.shuffle_lane_words3012();

  vec512_storage r{{}};

  for (size_t i = 0; i < 8; i++) {
    r.u32x16[i] = lo.u32x8[i];
  }
  for (size_t i = 0; i < 8; i++) {
    r.u32x16[8 + i] = hi.u32x8[i];
  }

  return r;
}

LIBC_INLINE constexpr vec512_storage
vec512_storage::shuffle_lane_words2301() const {
  vec256_storage lo{{}};
  vec256_storage hi{{}};

  for (size_t i = 0; i < 8; i++) {
    lo.u32x8[i] = this->u32x16[i];
  }
  for (size_t i = 0; i < 8; i++) {
    hi.u32x8[i] = this->u32x16[8 + i];
  }
  lo = lo.shuffle_lane_words2301();
  hi = hi.shuffle_lane_words2301();

  vec512_storage r{{}};

  for (size_t i = 0; i < 8; i++) {
    r.u32x16[i] = lo.u32x8[i];
  }
  for (size_t i = 0; i < 8; i++) {
    r.u32x16[8 + i] = hi.u32x8[i];
  }

  return r;
}

LIBC_INLINE constexpr vec512_storage
vec512_storage::shuffle_lane_words1230() const {
  vec256_storage lo{{}};
  vec256_storage hi{{}};

  for (size_t i = 0; i < 8; i++) {
    lo.u32x8[i] = this->u32x16[i];
  }
  for (size_t i = 0; i < 8; i++) {
    hi.u32x8[i] = this->u32x16[8 + i];
  }

  lo = lo.shuffle_lane_words1230();
  hi = hi.shuffle_lane_words1230();

  vec512_storage r{{}};

  for (size_t i = 0; i < 8; i++) {
    r.u32x16[i] = lo.u32x8[i];
  }
  for (size_t i = 0; i < 8; i++) {
    r.u32x16[8 + i] = hi.u32x8[i];
  }

  return r;
}

LIBC_INLINE constexpr cpp::array<vec512_storage, 4>
vec512_storage::transpose4(const vec512_storage &a, const vec512_storage &b,
                           const vec512_storage &c, const vec512_storage &d) {
  /*
   * a00:a01 a10:a11
   * b00:b01 b10:b11
   * c00:c01 c10:c11
   * d00:d01 d10:d11
   *       =>
   * a00:b00 c00:d00
   * a01:b01 c01:d01
   * a10:b10 c10:d10
   * a11:b11 c11:d11
   */
  vec256_storage a_lo{{}};
  vec256_storage b_lo{{}};

  for (size_t i = 0; i < 8; i++) {
    a_lo.u32x8[i] = a.u32x16[i];
  }
  for (size_t i = 0; i < 8; i++) {
    b_lo.u32x8[i] = b.u32x16[i];
  }
  auto const ab00 = immintrin::mm256_permute2x128_si256(a_lo, b_lo, 0x20);
  auto const ab01 = immintrin::mm256_permute2x128_si256(a_lo, b_lo, 0x31);

  vec256_storage a_hi{{}};
  vec256_storage b_hi{{}};

  for (size_t i = 0; i < 8; i++) {
    a_hi.u32x8[i] = a.u32x16[8 + i];
  }
  for (size_t i = 0; i < 8; i++) {
    b_hi.u32x8[i] = b.u32x16[8 + i];
  }
  auto const ab10 = immintrin::mm256_permute2x128_si256(a_hi, b_hi, 0x20);
  auto const ab11 = immintrin::mm256_permute2x128_si256(a_hi, b_hi, 0x31);

  vec256_storage c_lo{{}};
  vec256_storage d_lo{{}};

  for (size_t i = 0; i < 8; i++) {
    c_lo.u32x8[i] = c.u32x16[i];
  }
  for (size_t i = 0; i < 8; i++) {
    d_lo.u32x8[i] = d.u32x16[i];
  }
  auto const cd00 = immintrin::mm256_permute2x128_si256(c_lo, d_lo, 0x20);
  auto const cd01 = immintrin::mm256_permute2x128_si256(c_lo, d_lo, 0x31);

  vec256_storage c_hi{{}};
  vec256_storage d_hi{{}};

  for (size_t i = 0; i < 8; i++) {
    c_hi.u32x8[i] = c.u32x16[8 + i];
  }
  for (size_t i = 0; i < 8; i++) {
    d_hi.u32x8[i] = d.u32x16[8 + i];
  }
  auto const cd10 = immintrin::mm256_permute2x128_si256(c_hi, d_hi, 0x20);
  auto const cd11 = immintrin::mm256_permute2x128_si256(c_hi, d_hi, 0x31);

  auto r1 = vec512_storage::construct_from_vec256(ab00, cd00);
  auto r2 = vec512_storage::construct_from_vec256(ab01, cd01);
  auto r3 = vec512_storage::construct_from_vec256(ab10, cd10);
  auto r4 = vec512_storage::construct_from_vec256(ab11, cd11);

  return cpp::array<vec512_storage, 4>{r1, r2, r3, r4};
}

} // namespace vector_storage

} // namespace random

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL
