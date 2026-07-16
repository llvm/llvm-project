//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// VFABI symbol construction for mathvec routines.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MATHVEC_ABI_PREFIX_H
#define LLVM_LIBC_MATHVEC_ABI_PREFIX_H

#include "src/__support/common.h"
#include "src/__support/macros/properties/cpu_features.h"

// TODO: Implement a more intelligent solution for
// creating ABI symbols.
#ifndef LIBC_MATHVEC_FLOAT_VFABI_PREFIX
#if defined(LIBC_TARGET_CPU_HAS_AVX512F)
#define LIBC_MATHVEC_FLOAT_VFABI_PREFIX "_ZGVeN16v_"
#elif defined(LIBC_TARGET_CPU_HAS_AVX2)
#define LIBC_MATHVEC_FLOAT_VFABI_PREFIX "_ZGVdN8v_"
#elif defined(LIBC_TARGET_CPU_HAS_AVX)
#define LIBC_MATHVEC_FLOAT_VFABI_PREFIX "_ZGVcN8v_"
#elif defined(LIBC_TARGET_CPU_HAS_SSE2)
#define LIBC_MATHVEC_FLOAT_VFABI_PREFIX "_ZGVbN4v_"
#elif defined(LIBC_TARGET_CPU_HAS_ARM_NEON)
#define LIBC_MATHVEC_FLOAT_VFABI_PREFIX "_ZGVnN4v_"
#else
#error "Unsupported target for mathvec VFABI symbols"
#endif
#endif // LIBC_MATHVEC_FLOAT_VFABI_PREFIX

#define LIBC_VFABI_FLOAT_SYMBOL(name) LIBC_MATHVEC_FLOAT_VFABI_PREFIX #name

#endif // LLVM_LIBC_MATHVEC_ABI_PREFIX_H
