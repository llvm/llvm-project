//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_EXP_INTEGER_EVAL_H

#include "hdr/fenv_macros.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/compiler.h"
#include "src/__support/math/exp_double_eval.h"
#include "src/__support/uint128.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {
namespace exp_integer_internal {
// Q96 range reduction and Q64 evaluation. Every product fits UInt<128>;
// the final product is split into 1+u and 1+v terms to avoid overflow.
// GCC expands generic full-width multiplication before truncation. Clang's
// existing limb backend is already efficient; retain it to avoid a regression.
#if defined(LIBC_TYPES_HAS_INT128) && defined(LIBC_COMPILER_IS_GCC)
using UInt128 = ::UInt128;
#else
using UInt128 = UInt<128>;
#endif
LIBC_INLINE UInt128 zero() { return UInt128(0); }
LIBC_INLINE int length(const UInt128 *a) {
#if defined(LIBC_TYPES_HAS_INT128) && defined(LIBC_COMPILER_IS_GCC)
  uint64_t hi = uint64_t(*a >> 64);
  return hi ? 128 - cpp::countl_zero(hi) : 64 - cpp::countl_zero(uint64_t(*a));
#else
  return int(cpp::bit_width(*a));
#endif
}
LIBC_INLINE int cmp(const UInt128 *a, const UInt128 *b) {
  return *a < *b ? -1 : (*a > *b ? 1 : 0);
}
LIBC_INLINE UInt128 add(UInt128 a, UInt128 b) { return a + b; }
LIBC_INLINE UInt128 sub(UInt128 a, UInt128 b) { return a - b; }
LIBC_INLINE UInt128 inc(UInt128 a) { return a + UInt128(1); }
LIBC_INLINE UInt128 one_at(int n) {
  return n >= 0 && n < 128 ? UInt128(1) << size_t(n) : UInt128(0);
}
LIBC_INLINE bool nonzero(const UInt128 *a) { return *a != UInt128(0); }
LIBC_INLINE bool exceeds_bits(const UInt128 *a, int n) {
  if (n < 0)
    return true;
  if (n >= 128)
    return false;
  return *a >= one_at(n);
}
LIBC_INLINE unsigned bit(const UInt128 *a, int n) {
  return n >= 0 && n < 128 ? unsigned(uint64_t(*a >> size_t(n)) & 1) : 0;
}
LIBC_INLINE bool sticky(const UInt128 *a, int n) {
  if (n <= 0)
    return false;
  if (n >= 128)
    return *a != UInt128(0);
  return (*a & ((UInt128(1) << size_t(n)) - UInt128(1))) != UInt128(0);
}
LIBC_INLINE UInt128 shift(UInt128 a, int n, int up) {
  if (n >= 128)
    return UInt128(0);
  if (n <= -128)
    return UInt128(unsigned(up && a != UInt128(0)));
  if (n >= 0)
    return a << size_t(n);
  return (a >> size_t(-n)) + UInt128(unsigned(up && sticky(&a, -n)));
}
LIBC_INLINE UInt128 mul(UInt128 a, UInt128 b) { return a * b; }
// Callers bound (a,b) to (65,57) or (57,64) bits, so the rounded
// Q64 product fits uint64_t. Only the low-limb product needs 128 bits.
LIBC_INLINE UInt128 mul_q64(UInt128 a, uint64_t b, int up) {
  UInt128 product = UInt128(uint64_t(a)) * UInt128(b);
  uint64_t result = uint64_t(product >> 64) + uint64_t(a >> 64) * b;
  return UInt128(result + unsigned(up && uint64_t(product) != 0));
}
LIBC_ALWAYS_INLINE uint64_t round_binary64(UInt128 v, int q, int mode) {
  uint64_t high = uint64_t(v >> 64);
  // The scaled enclosure lies in [1/2,4); avoid bit scans on this common shape.
  int width = high < 4 && (high || uint64_t(v) >= (uint64_t(1) << 63))
                  ? 64 + int(high != 0) + int(high >= 2)
                  : length(&v);
  int top = width - 1 + q - 64;
  // The kernel deliberately accepts only normal outputs.
  if (top < -1022 || top > 1023)
    return 0;
  int k = 64 + top - 52 - q;
  if (k > 0 && k < 64) {
    uint64_t sig = uint64_t(v >> size_t(k));
    uint64_t discarded = uint64_t(v) & ((uint64_t(1) << k) - 1);
    bool up = mode == 1 && discarded != 0;
    if (mode == 0) {
      uint64_t halfway = uint64_t(1) << (k - 1);
      up = discarded > halfway || (discarded == halfway && (sig & 1));
    }
    sig += unsigned(up);
    if (sig & (uint64_t(1) << 53)) {
      sig >>= 1;
      ++top;
    }
    if (top > 1023)
      return 0;
    return (uint64_t(top + 1023) << 52) | (sig & 0x000fffffffffffffULL);
  }
  UInt128 sig = shift(v, -k, 0);
  bool up = mode == 1 && sticky(&v, k);
  if (mode == 0 && bit(&v, k - 1))
    up = sticky(&v, k - 1) || bit(&sig, 0);
  if (up)
    sig = inc(sig);
  if (bit(&sig, 53)) {
    sig = shift(sig, -1, 0);
    ++top;
  }
  if (top > 1023)
    return 0;
  return (uint64_t(top + 1023) << 52) | (uint64_t(sig) & 0x000fffffffffffffULL);
}
LIBC_ALWAYS_INLINE bool round_binary64_interval(UInt128 lo, UInt128 hi, int q,
                                                int mode, uint64_t *output) {
  uint64_t high = uint64_t(lo >> 64);
  uint64_t low_lo = uint64_t(lo), low_hi = uint64_t(hi);
  if (high == uint64_t(hi >> 64) && high < 4 &&
      (high ||
       (low_lo >= (uint64_t(1) << 63) && low_hi >= (uint64_t(1) << 63)))) {
    int width = 64 + int(high != 0) + int(high >= 2);
    int top = width - 1 + q - 64;
    if (top < -1022 || top > 1023)
      return false;
    int k = width - 53;
    uint64_t prefix = high << (64 - k);
    uint64_t lower = (low_lo >> k) | prefix;
    uint64_t upper = (low_hi >> k) | prefix;
    uint64_t mask = (uint64_t(1) << k) - 1;
    uint64_t discarded_lo = low_lo & mask, discarded_hi = low_hi & mask;
    if (mode == 0) {
      uint64_t halfway = uint64_t(1) << (k - 1);
      lower += unsigned(discarded_lo > halfway ||
                        (discarded_lo == halfway && (lower & 1)));
      upper += unsigned(discarded_hi > halfway ||
                        (discarded_hi == halfway && (upper & 1)));
    } else if (mode == 1) {
      lower += unsigned(discarded_lo != 0);
      upper += unsigned(discarded_hi != 0);
    }
    if (lower != upper)
      return false;
    // Addition propagates a rounded significand carry into the exponent.
    uint64_t bits =
        (uint64_t(top + 1023) << 52) + (lower - (uint64_t(1) << 52));
    if (bits >= UINT64_C(0x7ff0000000000000))
      return false;
    *output = bits;
    return true;
  }
  uint64_t lower = round_binary64(lo, q, mode);
  uint64_t upper = round_binary64(hi, q, mode);
  if (!lower || lower != upper)
    return false;
  *output = lower;
  return true;
}
LIBC_INLINE_VAR constexpr UInt128 LN2_LO =
    UInt128(0xd1cf79abc9e3b398ULL) | (UInt128(0xb17217f7ULL) << 64);
LIBC_INLINE_VAR constexpr UInt128 LOG2_E_63 = UInt128(0xb8aa3b295c17f0bbULL);
// Constants are generated with exact integer/rational arithmetic by
// libc/utils/mathtools/exp_integer_constants.py. For row i,
// T^128 <= 2^(128*64+i) < (T+1)^128; LN2_LO < 2^96*ln(2) < LN2_LO+1.
struct TableEntry {
  uint64_t lo, hi;
};
LIBC_INLINE_VAR constexpr TableEntry EXP2_TABLE[128] = {
    {0x0000000000000000ULL, 0x0000000000000001ULL},
    {0x0163da9fb33356d8ULL, 0x0000000000000001ULL},
    {0x02c9a3e778060ee6ULL, 0x0000000000000001ULL},
    {0x04315e86e7f84bd7ULL, 0x0000000000000001ULL},
    {0x059b0d31585743aeULL, 0x0000000000000001ULL},
    {0x0706b29ddf6ddc6dULL, 0x0000000000000001ULL},
    {0x0874518759bc808cULL, 0x0000000000000001ULL},
    {0x09e3ecac6f383452ULL, 0x0000000000000001ULL},
    {0x0b5586cf9890f629ULL, 0x0000000000000001ULL},
    {0x0cc922b7247f7407ULL, 0x0000000000000001ULL},
    {0x0e3ec32d3d1a2020ULL, 0x0000000000000001ULL},
    {0x0fb66affed31af23ULL, 0x0000000000000001ULL},
    {0x11301d0125b50a4eULL, 0x0000000000000001ULL},
    {0x12abdc06c31cbfb9ULL, 0x0000000000000001ULL},
    {0x1429aaea92ddfb34ULL, 0x0000000000000001ULL},
    {0x15a98c8a58e51248ULL, 0x0000000000000001ULL},
    {0x172b83c7d517adcdULL, 0x0000000000000001ULL},
    {0x18af9388c8de9bbbULL, 0x0000000000000001ULL},
    {0x1a35beb6fcb753cbULL, 0x0000000000000001ULL},
    {0x1bbe084045cd39abULL, 0x0000000000000001ULL},
    {0x1d4873168b9aa780ULL, 0x0000000000000001ULL},
    {0x1ed5022fcd91cb88ULL, 0x0000000000000001ULL},
    {0x2063b88628cd63b8ULL, 0x0000000000000001ULL},
    {0x21f49917ddc96255ULL, 0x0000000000000001ULL},
    {0x2387a6e75623866cULL, 0x0000000000000001ULL},
    {0x251ce4fb2a63f358ULL, 0x0000000000000001ULL},
    {0x26b4565e27cdd257ULL, 0x0000000000000001ULL},
    {0x284dfe1f5638096cULL, 0x0000000000000001ULL},
    {0x29e9df51fdee12c2ULL, 0x0000000000000001ULL},
    {0x2b87fd0dad98ffddULL, 0x0000000000000001ULL},
    {0x2d285a6e4030b400ULL, 0x0000000000000001ULL},
    {0x2ecafa93e2f5611cULL, 0x0000000000000001ULL},
    {0x306fe0a31b7152deULL, 0x0000000000000001ULL},
    {0x32170fc4cd831353ULL, 0x0000000000000001ULL},
    {0x33c08b26416ff4c9ULL, 0x0000000000000001ULL},
    {0x356c55f929ff0c94ULL, 0x0000000000000001ULL},
    {0x371a7373aa9caa71ULL, 0x0000000000000001ULL},
    {0x38cae6d05d86585aULL, 0x0000000000000001ULL},
    {0x3a7db34e59ff6ea1ULL, 0x0000000000000001ULL},
    {0x3c32dc313a8e4840ULL, 0x0000000000000001ULL},
    {0x3dea64c12342235bULL, 0x0000000000000001ULL},
    {0x3fa4504ac801ba0bULL, 0x0000000000000001ULL},
    {0x4160a21f72e29f84ULL, 0x0000000000000001ULL},
    {0x431f5d950a896dc7ULL, 0x0000000000000001ULL},
    {0x44e086061892d031ULL, 0x0000000000000001ULL},
    {0x46a41ed1d0057725ULL, 0x0000000000000001ULL},
    {0x486a2b5c13cd013cULL, 0x0000000000000001ULL},
    {0x4a32af0d7d3de672ULL, 0x0000000000000001ULL},
    {0x4bfdad5362a271d4ULL, 0x0000000000000001ULL},
    {0x4dcb299fddd0d63bULL, 0x0000000000000001ULL},
    {0x4f9b2769d2ca6ad3ULL, 0x0000000000000001ULL},
    {0x516daa2cf6641c11ULL, 0x0000000000000001ULL},
    {0x5342b569d4f81df0ULL, 0x0000000000000001ULL},
    {0x551a4ca5d920ec52ULL, 0x0000000000000001ULL},
    {0x56f4736b527da66eULL, 0x0000000000000001ULL},
    {0x58d12d497c7fd252ULL, 0x0000000000000001ULL},
    {0x5ab07dd48542958cULL, 0x0000000000000001ULL},
    {0x5c9268a5946b701cULL, 0x0000000000000001ULL},
    {0x5e76f15ad21486e9ULL, 0x0000000000000001ULL},
    {0x605e1b976dc08b07ULL, 0x0000000000000001ULL},
    {0x6247eb03a5584b1fULL, 0x0000000000000001ULL},
    {0x6434634ccc31fc76ULL, 0x0000000000000001ULL},
    {0x6623882552224912ULL, 0x0000000000000001ULL},
    {0x68155d44ca973081ULL, 0x0000000000000001ULL},
    {0x6a09e667f3bcc908ULL, 0x0000000000000001ULL},
    {0x6c012750bdabeed7ULL, 0x0000000000000001ULL},
    {0x6dfb23c651a2ef22ULL, 0x0000000000000001ULL},
    {0x6ff7df9519483cf8ULL, 0x0000000000000001ULL},
    {0x71f75e8ec5f73dd2ULL, 0x0000000000000001ULL},
    {0x73f9a48a58173bd5ULL, 0x0000000000000001ULL},
    {0x75feb564267c8bf6ULL, 0x0000000000000001ULL},
    {0x780694fde5d3f619ULL, 0x0000000000000001ULL},
    {0x7a11473eb0186d7dULL, 0x0000000000000001ULL},
    {0x7c1ed0130c1327c4ULL, 0x0000000000000001ULL},
    {0x7e2f336cf4e62105ULL, 0x0000000000000001ULL},
    {0x80427543e1a11b60ULL, 0x0000000000000001ULL},
    {0x82589994cce128acULL, 0x0000000000000001ULL},
    {0x8471a4623c7acce5ULL, 0x0000000000000001ULL},
    {0x868d99b4492ec80eULL, 0x0000000000000001ULL},
    {0x88ac7d98a6699665ULL, 0x0000000000000001ULL},
    {0x8ace5422aa0db5baULL, 0x0000000000000001ULL},
    {0x8cf3216b5448bef2ULL, 0x0000000000000001ULL},
    {0x8f1ae991577362b9ULL, 0x0000000000000001ULL},
    {0x9145b0b91ffc588aULL, 0x0000000000000001ULL},
    {0x93737b0cdc5e4f45ULL, 0x0000000000000001ULL},
    {0x95a44cbc8520ee9bULL, 0x0000000000000001ULL},
    {0x97d829fde4e4f8b9ULL, 0x0000000000000001ULL},
    {0x9a0f170ca07b9ba3ULL, 0x0000000000000001ULL},
    {0x9c49182a3f0901c7ULL, 0x0000000000000001ULL},
    {0x9e86319e32323182ULL, 0x0000000000000001ULL},
    {0xa0c667b5de564b29ULL, 0x0000000000000001ULL},
    {0xa309bec4a2d3358cULL, 0x0000000000000001ULL},
    {0xa5503b23e255c8b4ULL, 0x0000000000000001ULL},
    {0xa799e1330b3586f2ULL, 0x0000000000000001ULL},
    {0xa9e6b5579fdbf43eULL, 0x0000000000000001ULL},
    {0xac36bbfd3f379c0dULL, 0x0000000000000001ULL},
    {0xae89f995ad3ad5e8ULL, 0x0000000000000001ULL},
    {0xb0e07298db665908ULL, 0x0000000000000001ULL},
    {0xb33a2b84f15faf6bULL, 0x0000000000000001ULL},
    {0xb59728de559398e3ULL, 0x0000000000000001ULL},
    {0xb7f76f2fb5e46eaaULL, 0x0000000000000001ULL},
    {0xba5b030a10649840ULL, 0x0000000000000001ULL},
    {0xbcc1e904bc1d2247ULL, 0x0000000000000001ULL},
    {0xbf2c25bd71e08840ULL, 0x0000000000000001ULL},
    {0xc199bdd85529c222ULL, 0x0000000000000001ULL},
    {0xc40ab5fffd07a6d1ULL, 0x0000000000000001ULL},
    {0xc67f12e57d14b4a2ULL, 0x0000000000000001ULL},
    {0xc8f6d9406e7b511aULL, 0x0000000000000001ULL},
    {0xcb720dcef9069150ULL, 0x0000000000000001ULL},
    {0xcdf0b555dc3f9c44ULL, 0x0000000000000001ULL},
    {0xd072d4a07897b8d0ULL, 0x0000000000000001ULL},
    {0xd2f87080d89f18adULL, 0x0000000000000001ULL},
    {0xd5818dcfba48725dULL, 0x0000000000000001ULL},
    {0xd80e316c98397bb8ULL, 0x0000000000000001ULL},
    {0xda9e603db3285708ULL, 0x0000000000000001ULL},
    {0xdd321f301b4604b6ULL, 0x0000000000000001ULL},
    {0xdfc97337b9b5eb96ULL, 0x0000000000000001ULL},
    {0xe264614f5a128a12ULL, 0x0000000000000001ULL},
    {0xe502ee78b3ff6273ULL, 0x0000000000000001ULL},
    {0xe7a51fbc74c834b5ULL, 0x0000000000000001ULL},
    {0xea4afa2a490d9858ULL, 0x0000000000000001ULL},
    {0xecf482d8e67f08dbULL, 0x0000000000000001ULL},
    {0xefa1bee615a27771ULL, 0x0000000000000001ULL},
    {0xf252b376bba974e8ULL, 0x0000000000000001ULL},
    {0xf50765b6e4540674ULL, 0x0000000000000001ULL},
    {0xf7bfdad9cbe13891ULL, 0x0000000000000001ULL},
    {0xfa7c1819e90d82e9ULL, 0x0000000000000001ULL},
    {0xfd3c22b8f71f1097ULL, 0x0000000000000001ULL},
};
LIBC_INLINE_VAR constexpr UInt128 R_BOUND = UInt128(0x1642c8590b21642ULL);
LIBC_INLINE_VAR constexpr UInt128 INV_FACT_LO[7] = {
    UInt128(0x0000000000000000ULL) | (UInt128(0x1ULL) << 64),
    UInt128(0x0000000000000000ULL) | (UInt128(0x1ULL) << 64),
    UInt128(0x8000000000000000ULL) | (UInt128(0x0ULL) << 64),
    UInt128(0x2aaaaaaaaaaaaaaaULL) | (UInt128(0x0ULL) << 64),
    UInt128(0x0aaaaaaaaaaaaaaaULL) | (UInt128(0x0ULL) << 64),
    UInt128(0x0222222222222222ULL) | (UInt128(0x0ULL) << 64),
    UInt128(0x005b05b05b05b05bULL) | (UInt128(0x0ULL) << 64),
};
struct SignedValue {
  UInt128 mag;
  int neg;
};
LIBC_INLINE SignedValue difference(UInt128 a, UInt128 b) {
  SignedValue r;
  r.neg = cmp(&a, &b) < 0;
  r.mag = r.neg ? sub(b, a) : sub(a, b);
  return r;
}
template <int N, bool LowerNegative, bool UpperNegative>
LIBC_ALWAYS_INLINE bool horner_interval(uint64_t rlo, uint64_t rhi, UInt128 &lo,
                                        UInt128 &hi) {
  if (exceeds_bits(&hi, 65))
    return false;
  UInt128 cl = INV_FACT_LO[N], cu = N <= 2 ? cl : inc(cl), pl, pu;
  if constexpr (LowerNegative) {
    pl = mul_q64(hi, rlo, 1);
    if (cmp(&cl, &pl) < 0)
      return false;
    if constexpr (UpperNegative) {
      pu = mul_q64(lo, rhi, 0);
      if (cmp(&cu, &pu) < 0)
        return false;
      if constexpr (N >= 2) {
        lo = UInt128(uint64_t(cl) - uint64_t(pl));
        hi = UInt128(uint64_t(cu) - uint64_t(pu));
      } else {
        lo = sub(cl, pl);
        hi = sub(cu, pu);
      }
    } else {
      pu = mul_q64(hi, rhi, 1);
      if constexpr (N >= 2) {
        lo = UInt128(uint64_t(cl) - uint64_t(pl));
        hi = UInt128(uint64_t(cu) + uint64_t(pu));
      } else {
        lo = sub(cl, pl);
        hi = add(cu, pu);
      }
    }
  } else {
    pl = mul_q64(lo, rlo, 0);
    pu = mul_q64(hi, rhi, 1);
    if constexpr (N >= 2) {
      lo = UInt128(uint64_t(cl) + uint64_t(pl));
      hi = UInt128(uint64_t(cu) + uint64_t(pu));
    } else {
      lo = add(cl, pl);
      hi = add(cu, pu);
    }
  }
  if constexpr (N > 0)
    return horner_interval<N - 1, LowerNegative, UpperNegative>(rlo, rhi, lo,
                                                                hi);
  return true;
}
// A false return leaves output unchanged; the caller must use a complete
// fallback.
LIBC_INLINE bool exp64_candidate(uint64_t raw, int mode, uint64_t *output) {
  const int B = 64, C = 96;
  uint64_t abs = raw & UINT64_C(0x7fffffffffffffff);
  int neg = (int)(raw >> 63), ef = (int)(abs >> 52), scale, k, q, idx;
  UInt128 mag = zero(), xl, xu, kl, ku, lo, hi, tablo, tabhi, zl, zu;
  SignedValue rlo, rhi;
  uint64_t kmag;
  if (mode < 0 || mode > 3 || !output)
    return 0;
  /* Restrict to a range with normal outputs and away from the tiny branch.
   * All other cases go through the caller's established libc implementation. */
  if (abs < UINT64_C(0x3ca0000000000000) || abs >= UINT64_C(0x4085e00000000000))
    return 0; /* 700 */
  mag = UInt128((abs & 0x000fffffffffffffULL) | 0x0010000000000000ULL);
  scale = ef - 1023 - 52;
  /* Choose k by a cheap approximation; correctness does not require k to
   * equal the nearest integer, because the residual is enclosed separately. */
  // The accepted exponent range makes this a right shift of at least 99 bits.
  kmag =
      ef <= 1003 ? 0 : uint64_t(mul(mag, LOG2_E_63) >> 64) >> size_t(1067 - ef);
  if (kmag > UINT64_C(131072))
    return 0;
  k = neg ? -(int)kmag : (int)kmag;
  q = k / 128;
  idx = k % 128;
  if (idx < 0) {
    idx += 128;
    --q;
  }
  xl = shift(mag, C + scale, 0);
  xu = shift(mag, C + scale, 1);
  {
    UInt128 ki(kmag);
    UInt128 product = mul(ki, LN2_LO);
    kl = shift(product, -7, 0);
    ku = shift(add(product, ki), -7, 1);
  }
  rlo = difference(xl, ku);
  rhi = difference(xu, kl);
  if (neg) {
    SignedValue t = rlo;
    rlo = rhi;
    rhi = t;
    if (nonzero(&rlo.mag))
      rlo.neg = !rlo.neg;
    if (nonzero(&rhi.mag))
      rhi.neg = !rhi.neg;
  }
  /* Residual endpoints are on the Q96 grid. Convert outward to Q64. */
  rlo.mag = shift(rlo.mag, B - C, rlo.neg);
  rhi.mag = shift(rhi.mag, B - C, !rhi.neg);
  if (!nonzero(&rlo.mag))
    rlo.neg = 0;
  if (!nonzero(&rhi.mag))
    rhi.neg = 0;
  /* Degree-six Taylor polynomial, evaluated by interval Horner arithmetic.
   * With |r| <= 1/184, the absolute omitted tail is < one Q64 unit.
   * All Horner intermediates are positive; coefficients are enclosed exactly.
   */
  if (cmp(&rlo.mag, &R_BOUND) > 0 || cmp(&rhi.mag, &R_BOUND) > 0)
    return 0;
  lo = INV_FACT_LO[6];
  hi = inc(lo);
  bool polynomial_ok =
      rlo.neg ? (rhi.neg ? horner_interval<5, true, true>(
                               uint64_t(rlo.mag), uint64_t(rhi.mag), lo, hi)
                         : horner_interval<5, true, false>(
                               uint64_t(rlo.mag), uint64_t(rhi.mag), lo, hi))
              : horner_interval<5, false, false>(uint64_t(rlo.mag),
                                                 uint64_t(rhi.mag), lo, hi);
  if (!polynomial_ok)
    return false;
  lo = sub(lo, one_at(0));
  hi = inc(hi);
  tablo = UInt128(EXP2_TABLE[idx].lo) | (UInt128(EXP2_TABLE[idx].hi) << 64);
  tabhi = idx ? inc(tablo) : tablo;
  /* Split exp(r)=1+u and table=1+v to avoid a 130-bit product.
   * |u|<1/8, 0<=v<1; |u*v| needs at most 125 bits in Q128. */
  {
    UInt128 scale_unit = one_at(B), v = sub(tablo, scale_unit), u;
    if (cmp(&lo, &scale_unit) >= 0) {
      u = sub(lo, scale_unit);
      if (exceeds_bits(&u, 57))
        return 0;
      zl = add(add(tablo, u), mul_q64(u, uint64_t(v), 0));
    } else {
      u = sub(scale_unit, lo);
      if (exceeds_bits(&u, 57))
        return 0;
      zl = sub(sub(tablo, u), mul_q64(u, uint64_t(v), 1));
    }
    v = sub(tabhi, scale_unit);
    if (cmp(&hi, &scale_unit) >= 0) {
      u = sub(hi, scale_unit);
      if (exceeds_bits(&u, 57))
        return 0;
      zu = add(add(tabhi, u), mul_q64(u, uint64_t(v), 1));
    } else {
      u = sub(scale_unit, hi);
      if (exceeds_bits(&u, 57))
        return 0;
      zu = sub(sub(tabhi, u), mul_q64(u, uint64_t(v), 0));
    }
  }
  return round_binary64_interval(zl, zu, q, mode, output);
}

} // namespace exp_integer_internal
namespace integer_eval {
// Hybrid implementation: bounded integer computation with the existing
// implementation as a total fallback. No unresolved status escapes this API.
LIBC_INLINE double exp(double x) {
  uint64_t raw = cpp::bit_cast<uint64_t>(x);
  uint64_t abs = raw & 0x7fffffffffffffffULL;
  // Delegate the known ambiguous boundary without first evaluating the kernel.
  if (abs <= 0x3ca0000000000000ULL || abs >= 0x4085e00000000000ULL)
    return double_eval::exp(x);
  int r = fputil::quick_get_round();
  int mode = r == FE_UPWARD       ? 1
             : r == FE_DOWNWARD   ? 2
             : r == FE_TOWARDZERO ? 3
                                  : 0;
  uint64_t out;
  if (!exp_integer_internal::exp64_candidate(raw, mode, &out))
    return double_eval::exp(x);
  fputil::raise_except_if_required(FE_INEXACT);
  return cpp::bit_cast<double>(out);
}
} // namespace integer_eval
} // namespace math
} // namespace LIBC_NAMESPACE_DECL
#endif
