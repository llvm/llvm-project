/*===------------ larchintrin.h - LoongArch intrinsics ---------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef _LOONGARCH_BASE_INTRIN_H
#define _LOONGARCH_BASE_INTRIN_H

#ifdef __cplusplus
extern "C" {
#endif

#if __loongarch_grlen == 64
extern __inline int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __crc_w_d_w(long int _1, int _2) {
  return (int)__builtin_loongarch_crc_w_d_w((long int)_1, (int)_2);
}
#endif

#define __break(/*ui15*/ _1) __builtin_loongarch_break((_1))

#define __dbar(/*ui15*/ _1) __builtin_loongarch_dbar((_1))

#define __ibar(/*ui15*/ _1) __builtin_loongarch_ibar((_1))

#define __syscall(/*ui15*/ _1) __builtin_loongarch_syscall((_1))

#ifdef __cplusplus
}
#endif
#endif /* _LOONGARCH_BASE_INTRIN_H */
