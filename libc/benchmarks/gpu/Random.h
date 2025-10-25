//===-- Pseudo-random number generation utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_BENCHMARKS_GPU_RANDOM_H
#define LLVM_LIBC_BENCHMARKS_GPU_RANDOM_H

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"
#include "src/__support/sign.h"

namespace LIBC_NAMESPACE_DECL {
namespace benchmarks {

// Pseudo-random number generator (PRNG) that produces unsigned 64-bit, 32-bit,
// and 16-bit integers. The implementation is based on the xorshift* generator,
// seeded using SplitMix64 for robust initialization. For more details, see:
// https://en.wikipedia.org/wiki/Xorshift
class RandomGenerator {
  uint64_t state;

  static LIBC_INLINE uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = (x ^ (x >> 31));
    return x ? x : 0x9E3779B97F4A7C15ULL;
  }

public:
  explicit LIBC_INLINE RandomGenerator(uint64_t seed) noexcept
      : state(splitmix64(seed)) {}

  LIBC_INLINE uint64_t next64() noexcept {
    uint64_t x = state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state = x;
    return x * 0x2545F4914F6CDD1DULL;
  }

  LIBC_INLINE uint32_t next32() noexcept {
    return static_cast<uint32_t>(next64() >> 32);
  }

  LIBC_INLINE uint16_t next16() noexcept {
    return static_cast<uint16_t>(next64() >> 48);
  }
};

// Generates random floating-point numbers where the unbiased binary exponent
// is sampled uniformly in `[min_exp, max_exp]`. The significand bits are
// always randomized, while the sign is randomized by default but can be fixed.
// Evenly covers orders of magnitude; never yields Inf/NaN.
template <typename T> class UniformExponent {
  static_assert(cpp::is_same_v<T, float16> || cpp::is_same_v<T, float> ||
                    cpp::is_same_v<T, double>,
                "UniformExponent supports float16, float, and double");

  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using Storage = typename FPBits::StorageType;

public:
  explicit UniformExponent(int min_exp = -FPBits::EXP_BIAS,
                           int max_exp = FPBits::EXP_BIAS,
                           cpp::optional<Sign> forced_sign = cpp::nullopt)
      : min_exp(clamp_exponent(cpp::min(min_exp, max_exp))),
        max_exp(clamp_exponent(cpp::max(min_exp, max_exp))),
        forced_sign(forced_sign) {}

  LIBC_INLINE T operator()(RandomGenerator &rng) const noexcept {
    // Sample unbiased exponent e uniformly in [min_exp, max_exp] without modulo
    // bias, using rejection sampling
    auto sample_in_range = [&](uint64_t r) -> int32_t {
      const uint64_t range = static_cast<uint64_t>(
          static_cast<int64_t>(max_exp) - static_cast<int64_t>(min_exp) + 1);
      const uint64_t threshold = (-range) % range;
      while (r < threshold)
        r = rng.next64();
      return static_cast<int32_t>(min_exp + static_cast<int64_t>(r % range));
    };
    const int32_t e = sample_in_range(rng.next64());

    // Start from random bits to get random sign and mantissa
    FPBits xbits([&] {
      if constexpr (cpp::is_same_v<T, double>)
        return FPBits(rng.next64());
      else if constexpr (cpp::is_same_v<T, float>)
        return FPBits(rng.next32());
      else
        return FPBits(rng.next16());
    }());

    if (e == -FPBits::EXP_BIAS) {
      // Subnormal: biased exponent must be 0; ensure mantissa != 0 to avoid 0
      xbits.set_biased_exponent(Storage(0));
      if (xbits.get_mantissa() == Storage(0))
        xbits.set_mantissa(Storage(1));
    } else {
      // Normal: biased exponent in [1, 2 * FPBits::EXP_BIAS]
      const int32_t biased = e + FPBits::EXP_BIAS;
      xbits.set_biased_exponent(static_cast<Storage>(biased));
    }

    if (forced_sign)
      xbits.set_sign(*forced_sign);

    return xbits.get_val();
  }

private:
  static LIBC_INLINE int clamp_exponent(int val) noexcept {
    if (val < -FPBits::EXP_BIAS)
      return -FPBits::EXP_BIAS;

    if (val > FPBits::EXP_BIAS)
      return FPBits::EXP_BIAS;

    return val;
  }

  const int min_exp;
  const int max_exp;
  const cpp::optional<Sign> forced_sign;
};

// Generates random floating-point numbers that are uniformly distributed on
// a linear scale. Values are sampled from `[min_val, max_val)`.
template <typename T> class UniformLinear {
  static_assert(cpp::is_same_v<T, float16> || cpp::is_same_v<T, float> ||
                    cpp::is_same_v<T, double>,
                "UniformLinear supports float16, float, and double");

  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using Storage = typename FPBits::StorageType;

  static constexpr T MAX_NORMAL = FPBits::max_normal().get_val();

public:
  explicit UniformLinear(T min_val = -MAX_NORMAL, T max_val = MAX_NORMAL)
      : min_val(clamp_val(cpp::min(min_val, max_val))),
        max_val(clamp_val(cpp::max(min_val, max_val))) {}

  LIBC_INLINE T operator()(RandomGenerator &rng) const noexcept {
    double u = standard_uniform(rng.next64());
    double a = static_cast<double>(min_val);
    double b = static_cast<double>(max_val);
    double y = a + (b - a) * u;
    return static_cast<T>(y);
  }

private:
  static LIBC_INLINE T clamp_val(T val) noexcept {
    if (val < -MAX_NORMAL)
      return -MAX_NORMAL;

    if (val > MAX_NORMAL)
      return MAX_NORMAL;

    return val;
  }

  static LIBC_INLINE double standard_uniform(uint64_t x) noexcept {
    constexpr int PREC_BITS =
        LIBC_NAMESPACE::fputil::FPBits<double>::SIG_LEN + 1;
    constexpr int SHIFT_BITS = LIBC_NAMESPACE::fputil::FPBits<double>::EXP_LEN;
    constexpr double INV = 1.0 / static_cast<double>(1ULL << PREC_BITS);

    return static_cast<double>(x >> SHIFT_BITS) * INV;
  }

  const T min_val;
  const T max_val;
};

} // namespace benchmarks
} // namespace LIBC_NAMESPACE_DECL

#endif
