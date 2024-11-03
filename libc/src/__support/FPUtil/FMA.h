//===-- Common header for FMA implementations -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_FMA_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_FMA_H

#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/cpu_features.h" // LIBC_TARGET_CPU_HAS_FMA

#if defined(LIBC_TARGET_CPU_HAS_FMA)

#if defined(LIBC_TARGET_ARCH_IS_X86_64)
#include "x86_64/FMA.h"
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "aarch64/FMA.h"
#endif

#else
// FMA instructions are not available
#include "generic/FMA.h"
#include "src/__support/CPP/type_traits.h"

namespace __llvm_libc {
namespace fputil {

template <typename T> LIBC_INLINE T fma(T x, T y, T z) {
  return generic::fma(x, y, z);
}

} // namespace fputil
} // namespace __llvm_libc

#endif

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_FMA_H
