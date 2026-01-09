//===-- Fast random number gen method - wctype conversion -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_FASTRAND_H
#define LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_FASTRAND_H

#include "src/__support/wctype/conversion/utils/utils.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace random {

namespace fastrand {

// This seed value is very important for different inputs. Bad values are known
// to cause compilation errors and/or incorrect computations in some cases.
// Defaulted to `0xEF6F79ED30BA75A` at first, but this value is not sufficient.
// Candidate seeds are taken directly from the resulted computation of the
// original Rust code, this is important for reproduction until a parallel
// generator is written for LLVM libc!
// `0x64a727ea04c46a32` is another viable seed.
LIBC_INLINE_VAR constexpr uint64_t DEFAULT_RNG_SEED = 0xeec13c9f1362aa74;

class Rng {
public:
  LIBC_INLINE constexpr Rng() : Rng(DEFAULT_RNG_SEED) {}

  LIBC_INLINE constexpr Rng(uint64_t seed) : seed(seed) {}
  LIBC_INLINE constexpr Rng(const Rng &) = default;
  LIBC_INLINE constexpr Rng(Rng &&) = default;

  LIBC_INLINE constexpr Rng &operator=(const Rng &) = default;
  LIBC_INLINE constexpr Rng &operator=(Rng &&) = default;

  LIBC_INLINE constexpr uint64_t gen() const {
    constexpr uint64_t WY_CONST_0 = 0x2d35'8dcc'aa6c'78a5;
    constexpr uint64_t WY_CONST_1 = 0x8bb8'4b93'962e'acc9;

    auto s = conversion_utils::wrapping_add(this->seed, WY_CONST_0);
    this->seed = s;
    auto const t =
        static_cast<UInt128>(s) * static_cast<UInt128>(s ^ WY_CONST_1);
    return static_cast<uint64_t>(t) ^ static_cast<uint64_t>(t >> 64);
  }

  LIBC_INLINE constexpr uint8_t gen_byte() const {
    return static_cast<uint8_t>(this->gen());
  }

  LIBC_INLINE constexpr Rng fork() const { return Rng(this->gen()); }
  LIBC_INLINE constexpr void set(uint64_t i) const { this->seed = i; }

  LIBC_INLINE constexpr Rng replace(Rng &&n) const {
    auto ret = this->seed;
    this->seed = n.seed;
    return ret;
  }

private:
  mutable uint64_t seed;
};

} // namespace fastrand

} // namespace random

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_WCTYPE_CONVERSION_RANDOM_FASTRAND_H
