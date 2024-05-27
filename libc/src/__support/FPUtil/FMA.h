//===-- Common header for FMA implementations -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_FMA_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_FMA_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/cpu_features.h" // LIBC_TARGET_CPU_HAS_FMA

#if defined(LIBC_TARGET_CPU_HAS_FMA)

namespace LIBC_NAMESPACE {
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
} // namespace LIBC_NAMESPACE

#else
// FMA instructions are not available
#include "generic/FMA.h"

namespace LIBC_NAMESPACE {
namespace fputil {

template <typename T> LIBC_INLINE T fma(T x, T y, T z) {
  return generic::fma(x, y, z);
}

} // namespace fputil
} // namespace LIBC_NAMESPACE

#endif

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FMA_H
