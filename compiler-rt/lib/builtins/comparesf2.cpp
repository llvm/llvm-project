//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's float comparison routines
/// (__lesf2/__gesf2/__unordsf2 and their aliases) on top of LLVM-libc's shared
/// comparison builtins.
///
//===----------------------------------------------------------------------===//

#define SINGLE_PRECISION
#include "fp_lib.h"

#include "fp_compare_impl.inc"
#include "fp_libc_config.h"
#include "shared/builtins/gesf2.h"
#include "shared/builtins/lesf2.h"
#include "shared/builtins/unordsf2.h"

extern "C" {

COMPILER_RT_ABI CMP_RESULT __lesf2(fp_t a, fp_t b) {
  return LIBC_NAMESPACE::shared::lesf2(a, b);
}
#if defined(__ELF__)
COMPILER_RT_ALIAS(__lesf2, __cmpsf2)
#endif
COMPILER_RT_ALIAS(__lesf2, __eqsf2)
COMPILER_RT_ALIAS(__lesf2, __ltsf2)
COMPILER_RT_ALIAS(__lesf2, __nesf2)

COMPILER_RT_ABI CMP_RESULT __gesf2(fp_t a, fp_t b) {
  return LIBC_NAMESPACE::shared::gesf2(a, b);
}
COMPILER_RT_ALIAS(__gesf2, __gtsf2)

COMPILER_RT_ABI CMP_RESULT __unordsf2(fp_t a, fp_t b) {
  return LIBC_NAMESPACE::shared::unordsf2(a, b);
}

#if defined(__ARM_EABI__)
#if defined(COMPILER_RT_ARMHF_TARGET)
AEABI_RTABI int __aeabi_fcmpun(fp_t a, fp_t b) { return __unordsf2(a, b); }
#else
COMPILER_RT_ALIAS(__unordsf2, __aeabi_fcmpun)
#endif
#endif

} // extern "C"
