//===-- Single-precision SIMD e^x vector function -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/mathvec/expf.h"
#include "src/__support/common.h"
#include "src/__support/macros/properties/cpu_features.h"
#include "src/__support/mathvec/expf.h"

#ifndef LIBC_MATHVEC_EXPF_SYM
#if defined(LIBC_TARGET_CPU_HAS_AVX512F)
#define LIBC_MATHVEC_EXPF_SYM_PREFIX "_ZGVeN16v_"
#elif defined(LIBC_TARGET_CPU_HAS_AVX2)
#define LIBC_MATHVEC_EXPF_SYM_PREFIX "_ZGVdN8v_"
#elif defined(LIBC_TARGET_CPU_HAS_AVX)
#define LIBC_MATHVEC_EXPF_SYM_PREFIX "_ZGVcN8v_"
#elif defined(LIBC_TARGET_CPU_HAS_SSE2)
#define LIBC_MATHVEC_EXPF_SYM_PREFIX "_ZGVbN4v_"
#elif defined(LIBC_TARGET_CPU_HAS_ARM_NEON)
#define LIBC_MATHVEC_EXPF_SYM_PREFIX "_ZGVnN4v_"
#else
#define LIBC_MATHVEC_EXPF_SYM_PREFIX "__"
#endif // LIBC_TARGET_CPU_HAS_*

#define LIBC_MATHVEC_EXPF_SYM LIBC_MATHVEC_EXPF_SYM_PREFIX "expf"
#endif // LIBC_MATHVEC_EXPF_SYM

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(cpp::simd<float>, expf, (cpp::simd<float> x),
                   LIBC_MATHVEC_EXPF_SYM) {
  return mathvec::expf(x);
}

} // namespace LIBC_NAMESPACE_DECL
