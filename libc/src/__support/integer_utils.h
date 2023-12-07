//===-- Utilities for integers. ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_INTEGER_UTILS_H
#define LLVM_LIBC_SRC_SUPPORT_INTEGER_UTILS_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/common.h"

#include "builtin_wrappers.h"
#include "number_pair.h"

#include <stdint.h>

namespace __llvm_libc {

template <typename T> NumberPair<T> full_mul(T a, T b);

template <>
LIBC_INLINE NumberPair<uint32_t> full_mul<uint32_t>(uint32_t a, uint32_t b) {
  uint64_t prod = uint64_t(a) * uint64_t(b);
  NumberPair<uint32_t> result;
  result.lo = uint32_t(prod);
  result.hi = uint32_t(prod >> 32);
  return result;
}

template <>
LIBC_INLINE NumberPair<uint64_t> full_mul<uint64_t>(uint64_t a, uint64_t b) {
#ifdef __SIZEOF_INT128__
  __uint128_t prod = __uint128_t(a) * __uint128_t(b);
  NumberPair<uint64_t> result;
  result.lo = uint64_t(prod);
  result.hi = uint64_t(prod >> 64);
  return result;
#else
  NumberPair<uint64_t> pa = split(a);
  NumberPair<uint64_t> pb = split(b);
  NumberPair<uint64_t> prod;

  prod.lo = pa.lo * pb.lo;                           // exact
  prod.hi = pa.hi * pb.hi;                           // exact
  NumberPair<uint64_t> lo_hi = split(pa.lo * pb.hi); // exact
  NumberPair<uint64_t> hi_lo = split(pa.hi * pb.lo); // exact

  auto r1 = add_with_carry(prod.lo, lo_hi.lo << 32, uint64_t(0));
  prod.lo = r1.sum;
  prod.hi = add_with_carry(prod.hi, lo_hi.hi, r1.carry).sum;

  auto r2 = add_with_carry(prod.lo, hi_lo.lo << 32, uint64_t(0));
  prod.lo = r2.sum;
  prod.hi = add_with_carry(prod.hi, hi_lo.hi, r2.carry).sum;

  return prod;
#endif // __SIZEOF_INT128__
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_INTEGER_UTILS_H
