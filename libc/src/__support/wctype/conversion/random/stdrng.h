//===-- ChaCha12 PRNG for StdRng - wctype conversion ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_STDRNG_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_STDRNG_H

#include "imm.h"
#include "src/__support/wctype/conversion/utils/slice.h"
#include "src/__support/wctype/conversion/utils/utils.h"
#include "vec512_storage.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace random {

namespace chacha12 {

namespace {

LIBC_INLINE_VAR constexpr size_t BLOCK = 16;
LIBC_INLINE_VAR constexpr uint64_t BLOCK64 = BLOCK;
LIBC_INLINE_VAR constexpr uint64_t LOG2_BUFBLOCKS = 2;
LIBC_INLINE_VAR constexpr uint64_t BUFBLOCKS = 1 << LOG2_BUFBLOCKS;
LIBC_INLINE_VAR constexpr uint64_t BUFSZ64 = BLOCK64 * BUFBLOCKS;
LIBC_INLINE_VAR constexpr size_t BUFSZ = BUFSZ64;

struct ChaCha {
  vector_storage::vec128_storage b;
  vector_storage::vec128_storage c;
  vector_storage::vec128_storage d;
};

template <class V> struct State {
  V a, b, c, d;
};

template <typename V = vector_storage::vec512_storage>
LIBC_INLINE static constexpr State<V> round(State<V> x) {
  x.a += x.b;
  x.d = (x.d ^ x.a).rotate_each_word_right16();
  x.c += x.d;
  x.b = (x.b ^ x.c).rotate_each_word_right20();
  x.a += x.b;
  x.d = (x.d ^ x.a).rotate_each_word_right24();
  x.c += x.d;
  x.b = (x.b ^ x.c).rotate_each_word_right25();
  return x;
}

template <typename V = vector_storage::vec512_storage>
LIBC_INLINE static constexpr State<V> diagonalize(State<V> x) {
  x.b = x.b.shuffle_lane_words3012();
  x.c = x.c.shuffle_lane_words2301();
  x.d = x.d.shuffle_lane_words1230();
  return x;
}

template <typename V = vector_storage::vec512_storage>
LIBC_INLINE static constexpr State<V> undiagonalize(State<V> x) {
  x.b = x.b.shuffle_lane_words1230();
  x.c = x.c.shuffle_lane_words2301();
  x.d = x.d.shuffle_lane_words3012();
  return x;
}

LIBC_INLINE static constexpr vector_storage::vec128_storage
add_pos(vector_storage::vec128_storage &d, uint64_t i) {
  auto const d0 = d.u32x4;
  auto const incr =
      vector_storage::vec128_storage::from_lanes(cpp::array<uint64_t, 2>{i, 0});
  return immintrin::add_epi64(d0, incr.u32x4);
}

} // namespace

struct StdRng {
  mutable ChaCha core_state;
  mutable cpp::array<uint32_t, 64> results;
  mutable uint8_t index;

  LIBC_INLINE static constexpr cpp::array<uint8_t, 4> pcg32(uint64_t &state) {
    constexpr uint64_t MUL = 6364136223846793005ull;
    constexpr uint64_t INC = 11634580027462260723ull;

    state = conversion_utils::wrapping_mul(state, MUL) + INC;
    // Use PCG output function with to_le to generate x:
    uint32_t xorshifted = static_cast<uint32_t>(((state >> 18) ^ state) >> 27);
    uint32_t rot = state >> 59;
    uint32_t x = conversion_utils::rotate_right(xorshifted, rot);
    return conversion_utils::to_le_bytes(x);
  }

  LIBC_INLINE static constexpr StdRng from_seed(uint64_t state = 31415) {
    cpp::array<uint8_t, 32> key = {0};
    auto s = conversion_utils::Slice<uint8_t>(key.data(), 32);
    size_t chunk_size = 4;
    size_t num_chunks = s.size() / chunk_size;

    for (size_t i = 0; i < num_chunks; ++i) {
      auto chunk = conversion_utils::Slice<uint8_t>{s.data() + i * chunk_size,
                                                    chunk_size};
      cpp::array<uint8_t, 4> x = pcg32(state);
      uint8_t *dst = chunk.data();
      uint8_t *src = x.data();

      for (unsigned j = 0; j < chunk.size(); ++j)
        dst[j] = src[j];
    }
    auto key0 = vector_storage::vec128_storage::read_le(
        conversion_utils::Slice<uint8_t>(key.data(), key.size()).range(0, 16));
    auto key1 = vector_storage::vec128_storage::read_le(
        conversion_utils::Slice<uint8_t>(key.data(), key.size()).subspan(16));

    auto core_state = ChaCha{
        .b = key0,
        .c = key1,
        .d =
            vector_storage::vec128_storage(cpp::array<uint32_t, 4>{0, 0, 0, 0}),
    };
    auto results = cpp::array<uint32_t, 64>{0};
    uint8_t index = results.size();
    return StdRng{core_state, results, index};
  }

  LIBC_INLINE constexpr void generate_and_set(size_t index) const {
    LIBC_ASSERT(index < this->results.size());
    this->generate(this->results);
    this->index = static_cast<uint8_t>(index);
  }

  LIBC_INLINE static constexpr vector_storage::vec512_storage
  d0123(vector_storage::vec128_storage &d) {
    auto x1 = vector_storage::vec256_storage::construct_from_vec128(
        vector_storage::vec128_storage::from_lanes(
            cpp::array<uint64_t, 2>{0, 0}),
        vector_storage::vec128_storage::from_lanes(
            cpp::array<uint64_t, 2>{1, 0}));
    auto x2 = vector_storage::vec256_storage::construct_from_vec128(
        vector_storage::vec128_storage::from_lanes(
            cpp::array<uint64_t, 2>{2, 0}),
        vector_storage::vec128_storage::from_lanes(
            cpp::array<uint64_t, 2>{3, 0}));

    auto incr = vector_storage::vec512_storage::construct_from_vec256(x1, x2);

    vector_storage::vec128_storage p1, p2, p3, p4;
    for (size_t x = 0; x < 4; x++) {
      p1.u32x4[x] = incr.u32x16[x];
      p2.u32x4[x] = incr.u32x16[x + 4];
      p3.u32x4[x] = incr.u32x16[x + 8];
      p4.u32x4[x] = incr.u32x16[x + 12];
    }
    auto i = immintrin::mm_add_epi64(d, p1);
    auto j = immintrin::mm_add_epi64(d, p2);
    auto k = immintrin::mm_add_epi64(d, p3);
    auto l = immintrin::mm_add_epi64(d, p4);

    auto v = vector_storage::vec512_storage::new128(i, j, k, l);

    return v;
  }

  LIBC_INLINE constexpr void
  refill_wide_impl(uint32_t drounds, cpp::array<uint32_t, BUFSZ> &out) const {
    auto k = vector_storage::vec128_storage::from_lanes(
        {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574});
    vector_storage::vec128_storage b = core_state.b;
    vector_storage::vec128_storage c = core_state.c;
    auto x = State<vector_storage::vec512_storage>{
        .a = vector_storage::vec512_storage::construct_from_vec256(
            immintrin::mm256_setr_m128i(k, k),
            immintrin::mm256_setr_m128i(k, k)),
        .b = vector_storage::vec512_storage::construct_from_vec256(
            immintrin::mm256_setr_m128i(b, b),
            immintrin::mm256_setr_m128i(b, b)),
        .c = vector_storage::vec512_storage::construct_from_vec256(
            immintrin::mm256_setr_m128i(c, c),
            immintrin::mm256_setr_m128i(c, c)),
        .d = d0123(core_state.d),
    };

    for (size_t i = 0; i < drounds; i++) {
      x = round(x);
      x = undiagonalize(round(diagonalize(x)));
    }

    auto const kk = vector_storage::vec512_storage::construct_from_vec256(
        immintrin::mm256_setr_m128i(k, k), immintrin::mm256_setr_m128i(k, k));
    auto const sb1 = core_state.b;
    auto sb = vector_storage::vec512_storage::construct_from_vec256(
        immintrin::mm256_setr_m128i(sb1, sb1),
        immintrin::mm256_setr_m128i(sb1, sb1));
    auto const sc1 = core_state.c;
    auto sc = vector_storage::vec512_storage::construct_from_vec256(
        immintrin::mm256_setr_m128i(sc1, sc1),
        immintrin::mm256_setr_m128i(sc1, sc1));
    auto const sd = d0123(core_state.d);
    auto const &r = vector_storage::vec512_storage::transpose4(
        kk + x.a, x.b + sb, x.c + sc, x.d + sd);
    auto const ra = r[0];
    auto const rb = r[1];
    auto const rc = r[2];
    auto const rd = r[3];

    conversion_utils::Slice<uint32_t> sout(out.data(), out.size());
    sout.range(0, 16).copy_from_slice(ra.to_scalars());
    sout.range(16, 32).copy_from_slice(rb.to_scalars());
    sout.range(32, 48).copy_from_slice(rc.to_scalars());
    sout.range(48, 64).copy_from_slice(rd.to_scalars());
    vector_storage::vec128_storage rx{{}};
    auto tc = sd.to_lanes().u32x16;
    for (size_t z = 0; z < 4; z++) {
      rx.u32x4[z] = tc[z];
    }
    core_state.d = add_pos(rx, 4);
  }

  LIBC_INLINE constexpr void generate(cpp::array<uint32_t, BUFSZ> &out) const {
    refill_wide_impl(6, out);
  }

  LIBC_INLINE constexpr uint64_t next_u64() const {
    const auto read_u64 = [](conversion_utils::Slice<uint32_t> results,
                             size_t index) {
      auto data = results.range(index, index + 2);
      return (static_cast<uint64_t>(data[1]) << 32) |
             static_cast<uint64_t>(data[0]);
    };
    const auto len = this->results.size();
    const auto index = this->index;

    if (index < len - 1) {
      this->index += 2;
      // Read an u64 from the current index
      return read_u64(conversion_utils::Slice<uint32_t>(this->results.data(),
                                                        this->results.size()),
                      index);
    } else if (index >= len) {
      this->generate_and_set(2);
      return read_u64(conversion_utils::Slice<uint32_t>(this->results.data(),
                                                        this->results.size()),
                      0);
    } else {
      const uint64_t x = this->results[len - 1];
      this->generate_and_set(1);
      const uint64_t y = this->results[0];
      return (y << 32) | x;
    }
  }

  LIBC_INLINE constexpr uint64_t random() const { return next_u64(); }
};

} // namespace chacha12

} // namespace random

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_RAND_COMMON_H
