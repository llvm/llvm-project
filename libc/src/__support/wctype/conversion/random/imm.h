//===-- Portable subset of <immintrin.h> ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Only little-endian is supported (runtime code is not affected by this).

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_IMM_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_IMM_H

#include "vec512_storage.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace random {

namespace immintrin {

using random::vector_storage::vec128_storage;
using random::vector_storage::vec256_storage;
using random::vector_storage::vec512_storage;

LIBC_INLINE static constexpr vec256_storage
mm256_add_epi32(const vec256_storage &a, const vec256_storage &b) {
  vec256_storage r{{}};
  for (int i = 0; i < 8; ++i) {
    r.u32x8[i] = a.u32x8[i] + b.u32x8[i]; // modulo 2^32
  }
  return r;
}

LIBC_INLINE static constexpr vec512_storage
mm256_add_epi32(const vec512_storage &a, const vec512_storage &b) {
  vec512_storage r{{}};
  for (int i = 0; i < 16; ++i) {
    r.u32x16[i] = a.u32x16[i] + b.u32x16[i]; // modulo 2^32
  }
  return r;
}

LIBC_INLINE static constexpr vec256_storage
mm256_xor_si256(const cpp::array<uint32_t, 8> &a,
                const cpp::array<uint32_t, 8> &b) {
  cpp::array<uint32_t, 8> r{};
  for (int i = 0; i < 8; ++i) {
    r[i] = a[i] ^ b[i];
  }
  return r;
}

LIBC_INLINE static constexpr vec512_storage
mm256_xor_si256(const cpp::array<uint32_t, 16> &a,
                const cpp::array<uint32_t, 16> &b) {
  vec512_storage r{.u32x16 = {}};
  for (int i = 0; i < 16; ++i) {
    r.u32x16[i] = a[i] ^ b[i];
  }
  return r;
}

LIBC_INLINE static constexpr vec256_storage
mm256_shuffle_epi8(const vec256_storage &a, const vec256_storage &b) {
  vec256_storage r{{}};
  for (size_t k = 0; k < 8; k++) {
    r.u32x8[k] = 0;
  }

  // Helper for 128-bit lane (16 bytes)
  auto shuffle_128 = [](const uint32_t *src, const uint32_t *ctrl,
                        uint32_t *dst) {
    // dst must be zero-initialized by caller
    for (int i = 0; i < 16; ++i) {
      uint8_t c = (ctrl[i / 4] >> ((i % 4) * 8)) & 0xFF;

      if (c & 0x80) {
        // zero byte → already zero
        continue;
      }

      int k = c & 0x0F;
      uint8_t byte = (src[k / 4] >> ((k % 4) * 8)) & 0xFF;

      dst[i / 4] |= static_cast<uint32_t>(byte) << ((i % 4) * 8);
    }
  };

  // Shuffle lower 128-bit lane
  shuffle_128(&a.u32x8[0], &b.u32x8[0], &r.u32x8[0]);
  // Shuffle upper 128-bit lane
  shuffle_128(&a.u32x8[4], &b.u32x8[4], &r.u32x8[4]);

  return r;
}

LIBC_INLINE static constexpr vec256_storage
mm256_set_epi64x(long long a, long long b, long long c, long long d) {
  vec256_storage v{{}};

  // Lower 128-bit lane (d, c)
  v.u32x8[0] = static_cast<uint32_t>(d);       // d[31:0]
  v.u32x8[1] = static_cast<uint32_t>(d >> 32); // d[63:32]
  v.u32x8[2] = static_cast<uint32_t>(c);       // c[31:0]
  v.u32x8[3] = static_cast<uint32_t>(c >> 32); // c[63:32]

  // Upper 128-bit lane (b, a)
  v.u32x8[4] = static_cast<uint32_t>(b);       // b[31:0]
  v.u32x8[5] = static_cast<uint32_t>(b >> 32); // b[63:32]
  v.u32x8[6] = static_cast<uint32_t>(a);       // a[31:0]
  v.u32x8[7] = static_cast<uint32_t>(a >> 32); // a[63:32]

  return v;
}

LIBC_INLINE static constexpr vec256_storage
mm256_or_si256(const vec256_storage &a, const vec256_storage &b) {
  vec256_storage r{{}};
  for (int i = 0; i < 8; ++i) {
    r.u32x8[i] = a.u32x8[i] | b.u32x8[i];
  }
  return r;
}

LIBC_INLINE static constexpr vec256_storage
mm256_srli_epi32(const vec256_storage &a, int count) {
  vec256_storage r{{}};

  // Cap the shift count at 31, as larger shifts produce zero
  const int c = count & 0x1F;

  for (int i = 0; i < 8; ++i) {
    r.u32x8[i] = a.u32x8[i] >> c;
  }

  return r;
}

LIBC_INLINE static constexpr vec256_storage
mm256_slli_epi32(const vec256_storage &a, int count) {
  vec256_storage r{{}};

  // Cap the shift count at 31, as larger shifts produce zero
  const int c = count & 0x1F;

  for (int i = 0; i < 8; ++i) {
    r.u32x8[i] = a.u32x8[i] << c;
  }

  return r;
}

LIBC_INLINE static constexpr vec256_storage
mm256_permute2x128_si256(const vec256_storage &V1, const vec256_storage &V2,
                         int M) {
  vec256_storage r{{}};

  // For each 128-bit destination half
  for (int half = 0; half < 2; ++half) {
    int control = (M >> (half * 4)) & 0xF;
    int dst_base = half * 4;

    if (control & 0x8) {
      // bit 3 set → zero this 128-bit half
      for (int i = 0; i < 4; ++i) {
        r.u32x8[dst_base + i] = 0;
      }
    } else {
      // bits [1:0] select source half
      const vec256_storage *src{};
      int src_base{};

      switch (control & 0x3) {
      case 0: // V1 lower
        src = &V1;
        src_base = 0;
        break;
      case 1: // V1 upper
        src = &V1;
        src_base = 4;
        break;
      case 2: // V2 lower
        src = &V2;
        src_base = 0;
        break;
      case 3: // V2 upper
        src = &V2;
        src_base = 4;
        break;
      }

      for (int i = 0; i < 4; ++i) {
        r.u32x8[dst_base + i] = src->u32x8[src_base + i];
      }
    }
  }

  return r;
}

// a_lo and a_hi are each 128-bit vectors represented as 4 x 32-bit integers
LIBC_INLINE static constexpr vec256_storage
mm256_setr_m128i(const vec128_storage &lo, const vec128_storage &hi) {
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

LIBC_INLINE static constexpr vec256_storage
mm256_shuffle_epi32(vec256_storage a, int imm) {
  vec256_storage r{{}};

  // lower half (elements 0..3)
  for (int i = 0; i < 4; ++i) {
    int src = (imm >> (2 * i)) & 0x3;
    r.u32x8[i] = a.u32x8[src];
  }

  // upper half (elements 4..7)
  for (int i = 0; i < 4; ++i) {
    int src = (imm >> (2 * i)) & 0x3;
    r.u32x8[4 + i] = a.u32x8[4 + src];
  }

  return r;
}

LIBC_INLINE static constexpr vec128_storage
mm256_extracti128_si256(const vec256_storage &V, int M) {
  const int base = (M & 1) * 4;
  return {{V.u32x8[base + 0], V.u32x8[base + 1], V.u32x8[base + 2],
           V.u32x8[base + 3]}};
}

LIBC_INLINE static constexpr vec128_storage
mm_add_epi64(const cpp::array<uint32_t, 4> &a,
             const cpp::array<uint32_t, 4> &b) {
  return {cpp::array<uint32_t, 4>{
      a[0] + b[0],
      a[1] + b[1],
      a[2] + b[2],
      a[3] + b[3],
  }};
}

LIBC_INLINE static constexpr cpp::array<uint32_t, 4>
add_epi64(const cpp::array<uint32_t, 4> &a, const cpp::array<uint32_t, 4> &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
}

} // namespace immintrin

} // namespace random

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_IMM_H
