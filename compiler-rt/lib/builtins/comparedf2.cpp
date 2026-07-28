//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's double comparison routines
/// (__ledf2/__gedf2/__unorddf2 and their aliases) on top of LLVM-libc's shared
/// comparison builtins.
///
//===----------------------------------------------------------------------===//

#define DOUBLE_PRECISION
#include "fp_lib.h"

#include "fp_compare_impl.inc"
#include "fp_libc_config.h"
#include "shared/builtins/gedf2.h"
#include "shared/builtins/ledf2.h"
#include "shared/builtins/unorddf2.h"

extern "C" {

COMPILER_RT_ABI CMP_RESULT __ledf2(fp_t a, fp_t b) {
  return LIBC_NAMESPACE::shared::ledf2(a, b);
}
#if defined(__ELF__)
COMPILER_RT_ALIAS(__ledf2, __cmpdf2)
#endif
COMPILER_RT_ALIAS(__ledf2, __eqdf2)
COMPILER_RT_ALIAS(__ledf2, __ltdf2)
COMPILER_RT_ALIAS(__ledf2, __nedf2)

COMPILER_RT_ABI CMP_RESULT __gedf2(fp_t a, fp_t b) {
  return LIBC_NAMESPACE::shared::gedf2(a, b);
}
COMPILER_RT_ALIAS(__gedf2, __gtdf2)

COMPILER_RT_ABI CMP_RESULT __unorddf2(fp_t a, fp_t b) {
  return LIBC_NAMESPACE::shared::unorddf2(a, b);
}

#if defined(__ARM_EABI__)
#if defined(COMPILER_RT_ARMHF_TARGET)
AEABI_RTABI int __aeabi_dcmpun(fp_t a, fp_t b) { return __unorddf2(a, b); }
#else
COMPILER_RT_ALIAS(__unorddf2, __aeabi_dcmpun)
#endif
#endif

} // extern "C"
