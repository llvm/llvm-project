/* ===-- fixunsdfsi.c - Implement __fixunsdfsi -----------------------------===
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
typedef su_int fixuint_t;
#include "fp_fixuint_impl.inc"

COMPILER_RT_ABI su_int
__fixunsdfsi(fp_t a) {
    return __fixuint(a);
}

#if defined(__ARM_EABI__)
AEABI_RTABI su_int __aeabi_d2uiz(fp_t a) {
  return __fixunsdfsi(a);
}
#endif

