//===-- Utilities for integers. ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_INTEGER_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_INTEGER_UTILS_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/common.h"

#include "math_extras.h"
#include "number_pair.h"

#include <stdint.h>

namespace LIBC_NAMESPACE {

template <typename T> NumberPair<T> full_mul(T a, T b) {
  NumberPair<T> pa = split(a);
  NumberPair<T> pb = split(b);
  NumberPair<T> prod;

  prod.lo = pa.lo * pb.lo;                    // exact
  prod.hi = pa.hi * pb.hi;                    // exact
  NumberPair<T> lo_hi = split(pa.lo * pb.hi); // exact
  NumberPair<T> hi_lo = split(pa.hi * pb.lo); // exact

  constexpr size_t HALF_BIT_WIDTH = sizeof(T) * CHAR_BIT / 2;

  auto r1 = add_with_carry(prod.lo, lo_hi.lo << HALF_BIT_WIDTH, T(0));
  prod.lo = r1.sum;
  prod.hi = add_with_carry(prod.hi, lo_hi.hi, r1.carry).sum;

  auto r2 = add_with_carry(prod.lo, hi_lo.lo << HALF_BIT_WIDTH, T(0));
  prod.lo = r2.sum;
  prod.hi = add_with_carry(prod.hi, hi_lo.hi, r2.carry).sum;

  return prod;
}

template <>
LIBC_INLINE NumberPair<uint32_t> full_mul<uint32_t>(uint32_t a, uint32_t b) {
  uint64_t prod = uint64_t(a) * uint64_t(b);
  NumberPair<uint32_t> result;
  result.lo = uint32_t(prod);
  result.hi = uint32_t(prod >> 32);
  return result;
}

#ifdef __SIZEOF_INT128__
template <>
LIBC_INLINE NumberPair<uint64_t> full_mul<uint64_t>(uint64_t a, uint64_t b) {
  __uint128_t prod = __uint128_t(a) * __uint128_t(b);
  NumberPair<uint64_t> result;
  result.lo = uint64_t(prod);
  result.hi = uint64_t(prod >> 64);
  return result;
}
#endif // __SIZEOF_INT128__

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_INTEGER_UTILS_H
