//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integer-only implementation of exp.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H

#include "hdr/fenv_macros.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/frac128.h"
#include "src/__support/frac64.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/exp_integer_utils.h"

// TODO: share *_integer_utils.h with exp2 the same way we did with expf/exp2f.

namespace LIBC_NAMESPACE_DECL {
namespace math {
namespace static_rounding {

LIBC_ALWAYS_INLINE uint64_t exp_bits(uint64_t x_u,
                                     [[maybe_unused]] int rounding) {
  using FPBits = fputil::FPBits<double>;

  uint64_t x_abs = x_u & 0x7fff'ffff'ffff'ffff;

  if (x_abs > FPBits::inf().uintval())
    return x_u | (uint64_t(1) << (FPBits::FRACTION_LEN - 1));

  if (LIBC_UNLIKELY(
          x_u >= 0xc087'4910'd52d'3052 ||
          (x_u <= 0xbca0'0000'0000'0000 && x_u >= 0x4086'2e42'fefa'39f0) ||
          x_u <= 0x3ca0'0000'0000'0000)) {
    // |x| <= 2^-53
    if (x_abs <= 0x3ca0'0000'0000'0000) {
      constexpr uint64_t ONE = FPBits::one().uintval();

      if (x_abs == 0)
        return ONE;

      bool is_neg = (x_u >> 63) != 0;

      if (x_abs == 0x3ca0'0000'0000'0000) {
        if (rounding == FE_TONEAREST || (rounding == FE_UPWARD && !is_neg) ||
            ((rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO) && is_neg))
          return is_neg ? ONE - 1 : ONE + 1;
      }

#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
      if (rounding == FE_UPWARD && !is_neg)
        return ONE + 1;
      if ((rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO) && is_neg)
        return ONE - 1;
#endif

      return ONE;
    }

    // x <= log(2^-1075) || x >= 0x1.6232bdd7abcd3p+9 or inf/nan.

    // x <= log(2^-1075) or -inf/nan
    if (x_u >= 0xc087'4910'd52d'3052) {
      // exp(-Inf) = 0
      if (x_abs == FPBits::inf().uintval())
        return FPBits::zero().uintval();

#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
      if (rounding == FE_UPWARD)
        return FPBits::min_subnormal().uintval();
#endif
      return FPBits::zero().uintval();
    }

    // x >= round(log(MAX_NORMAL), D, RU) = 0x1.62e42fefa39fp+9 or +inf/nan
    // x is finite
    if (x_u < 0x7ff0'0000'0000'0000) {
#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
      if (rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO)
        return FPBits::max_normal().uintval();
#endif
    }

    // x is +inf or nan
    return FPBits::inf().uintval();
  }

  // Main calculations

  bool is_neg = (x_u >> 63) != 0;
  uint32_t x_e = static_cast<uint32_t>(x_abs >> FPBits::FRACTION_LEN);
  uint64_t x_s = x_abs & FPBits::FRACTION_MASK;

  // The idea of the algorithm is that:
  // Let:
  //  - y = x / ln(2) --> exp(x) = 2^y.
  //  - y = hi + mid + low
  //
  // With:
  //  - hi is an integer
  //  - mid * 2^4 is an integer
  //  - lo are the remainder bits
  //
  // Then:
  //  exp(x) = 2^hi * 2^mid * 2^lo
  //
  // With this formula:
  //  - Multiplying by 2^hi is exact and cheap, via adding into the exponent
  //  field
  //  - 2^mid can be calculated via the LUT declared above
  //  - 2^lo ~ 1 + lo + a0 * lo^2 + ...
  //
  // Then we can construct exp(x) pretty easily, as hi, mid, lo bits can be
  // separate and be used independently, then we only need to reconstruct in the
  // final steps, which makes our life easier.
  //
  // This algorithm will be implemented in 2 paths: fast (64-bit) and accurate
  // (128-bit) paths. The accurate path will only kicks in when it's hard to
  // round.

  // Range reduction

  // binary64 recall:
  // 1 sign bit
  // 11 exp bits
  // 52 sig bits

  uint64_t x_s_shifted = (x_s << 11) | (1ull << 63);

  int x_e_unbiased = static_cast<int>(x_e) - FPBits::EXP_BIAS;
  Frac128 l2y_r;
  uint64_t e_y;
  int d;

  exp_scale(x_s_shifted, x_e_unbiased, is_neg, l2y_r, e_y, d);

  // d >= 53 --> k >= 1076
  // --> guaranteed to be below 2^-1074
  //
  // underflow
  if (LIBC_UNLIKELY(is_neg && d >= 53)) {
    return 0.0;
  }

  Frac64 result = exp_fast_path(l2y_r.val[1]);

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  return exp_handle_rounding(result, is_neg, d, e_y, rounding);
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

  unsigned shift = is_neg && d >= 0 ? static_cast<unsigned>(12 + d) : 11;
  uint64_t mask = UINT64_MAX >> (64 - shift);

  constexpr uint64_t ROUNDING_ERROR = 32;
  uint64_t tail = result.val[0];
  if (rounding == FE_TONEAREST)
    tail += uint64_t(1) << (shift - 1);
  bool is_hard = ((tail + ROUNDING_ERROR) & mask) <= 2 * ROUNDING_ERROR;

  if (LIBC_LIKELY(!is_hard)) {
    return exp_handle_rounding(result, is_neg, d, e_y, rounding);
  }

  // Dial back to 128-bit path for hard-to-round cases
  return exp_accurate_path(l2y_r, is_neg, d, e_y, rounding);
}

LIBC_ALWAYS_INLINE double exp(double x, [[maybe_unused]] int rounding) {
  return cpp::bit_cast<double>(exp_bits(cpp::bit_cast<uint64_t>(x), rounding));
}

} // namespace static_rounding

namespace integer_eval {

LIBC_ALWAYS_INLINE double exp(double x) {
  return static_rounding::exp(x, FE_TONEAREST);
}

} // namespace integer_eval
} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H
