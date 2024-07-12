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
#include "src/__support/FPUtil/generic/FMA.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/cpu_features.h" // LIBC_TARGET_CPU_HAS_FMA

namespace LIBC_NAMESPACE {
namespace fputil {

template <typename OutType, typename InType>
LIBC_INLINE OutType fma(InType x, InType y, InType z) {
  return generic::fma<OutType>(x, y, z);
}

#ifdef LIBC_TARGET_CPU_HAS_FMA
template <> LIBC_INLINE float fma(float x, float y, float z) {
  return __builtin_fmaf(x, y, z);
}

template <> LIBC_INLINE double fma(double x, double y, double z) {
  return __builtin_fma(x, y, z);
}
#endif // LIBC_TARGET_CPU_HAS_FMA

} // namespace fputil
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_FMA_H
