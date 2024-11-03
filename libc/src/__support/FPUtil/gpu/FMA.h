//===-- GPU implementations of the fma function -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_GPU_FMA_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_GPU_FMA_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/config.h"

// These intrinsics map to the FMA instrunctions in the target ISA for the GPU.
// The default rounding mode generated from these will be to the nearest even.
static_assert(LIBC_HAS_BUILTIN(__builtin_fma), "FMA builtins must be defined");
static_assert(LIBC_HAS_BUILTIN(__builtin_fmaf), "FMA builtins must be defined");

namespace __llvm_libc {
namespace fputil {

template <typename T>
LIBC_INLINE cpp::enable_if_t<cpp::is_same_v<T, float>, T> fma(T x, T y, T z) {
  return __builtin_fmaf(x, y, z);
}

template <typename T>
LIBC_INLINE cpp::enable_if_t<cpp::is_same_v<T, double>, T> fma(T x, T y, T z) {
  return __builtin_fma(x, y, z);
}

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_GPU_FMA_H
