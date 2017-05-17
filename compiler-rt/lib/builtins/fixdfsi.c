/* ===-- fixdfsi.c - Implement __fixdfsi -----------------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 */

#define DOUBLE_PRECISION
#include "fp_lib.h"
typedef si_int fixint_t;
typedef su_int fixuint_t;
#include "fp_fixint_impl.inc"

COMPILER_RT_ABI si_int
__fixdfsi(fp_t a) {
    return __fixint(a);
}

#if defined(__ARM_EABI__)
AEABI_RTABI si_int __aeabi_d2iz(fp_t a) {
  return __fixdfsi(a);
}
#endif

