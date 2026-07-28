//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's float128 comparison routines
/// (__letf2/__getf2/__unordtf2 and their aliases) on top of LLVM-libc's shared
/// comparison builtins.
///
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"

#if defined(CRT_HAS_TF_MODE)
#include "fp_compare_impl.inc"
#include "fp_libc_config.h"
#include "shared/builtins/getf2.h"
#include "shared/builtins/letf2.h"
#include "shared/builtins/unordtf2.h"

extern "C" {

COMPILER_RT_ABI CMP_RESULT __letf2(fp_t a, fp_t b) {
  return LIBC_NAMESPACE::shared::letf2(a, b);
}
#if defined(__ELF__)
COMPILER_RT_ALIAS(__letf2, __cmptf2)
#endif
COMPILER_RT_ALIAS(__letf2, __eqtf2)
COMPILER_RT_ALIAS(__letf2, __lttf2)
COMPILER_RT_ALIAS(__letf2, __netf2)

COMPILER_RT_ABI CMP_RESULT __getf2(fp_t a, fp_t b) {
  return LIBC_NAMESPACE::shared::getf2(a, b);
}
COMPILER_RT_ALIAS(__getf2, __gttf2)

COMPILER_RT_ABI CMP_RESULT __unordtf2(fp_t a, fp_t b) {
  return LIBC_NAMESPACE::shared::unordtf2(a, b);
}

} // extern "C"

#endif
