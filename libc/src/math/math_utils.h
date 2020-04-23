//===-- Collection of utils for implementing math functions -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_MATH_UTILS_H
#define LLVM_LIBC_SRC_MATH_MATH_UTILS_H

#include "include/errno.h"
#include "include/math.h"

#include "src/__support/common.h"
#include "src/errno/llvmlibc_errno.h"

#include <stdint.h>

namespace __llvm_libc {

static inline float with_errnof(float x, int err) {
  if (math_errhandling & MATH_ERRNO)
    llvmlibc_errno = err;
  return x;
}

static inline uint32_t as_uint32_bits(float x) {
  return *reinterpret_cast<uint32_t *>(&x);
}

static inline float as_float(uint32_t x) {
  return *reinterpret_cast<float *>(&x);
}

static inline double as_double(uint64_t x) {
  return *reinterpret_cast<double *>(&x);
}

static inline constexpr float invalidf(float x) {
  float y = (x - x) / (x - x);
  return isnan(x) ? y : with_errnof(y, EDOM);
}

static inline void force_eval_float(float x) { volatile float y UNUSED = x; }

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_MATH_UTILS_H
