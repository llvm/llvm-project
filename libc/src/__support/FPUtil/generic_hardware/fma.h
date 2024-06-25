//===-- Generic hardware implementation of fused multiply-add ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_SRC___SUPPORT_FPUTIL_GENERIC_HARDWARE_FMA_H
#define LIBC_SRC___SUPPORT_FPUTIL_GENERIC_HARDWARE_FMA_H

#include "src/__support/common.h"
#include "src/__support/macros/properties/cpu_features.h"

namespace LIBC_NAMESPACE::fputil::generic_hardware {

#ifdef LIBC_TARGET_CPU_HAS_FMA
LIBC_INLINE float fma(float x, float y, float z) {
  return __builtin_fmaf(x, y, z);
}

LIBC_INLINE double fma(double x, double y, double z) {
  return __builtin_fma(x, y, z);
}
#endif // LIBC_TARGET_CPU_HAS_FMA

} // namespace LIBC_NAMESPACE::fputil::generic_hardware

#endif // LIBC_SRC___SUPPORT_FPUTIL_GENERIC_HARDWARE_FMA_H
