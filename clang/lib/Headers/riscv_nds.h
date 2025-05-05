/*===---- riscv_nds.h - Andes intrinsics -----------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __RISCV_NDS_H
#define __RISCV_NDS_H

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__riscv_xandesperf)

#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__))

static __inline__ long __DEFAULT_FN_ATTRS __riscv_nds_ffb(unsigned long a,
                                                          unsigned long b) {
  return __builtin_riscv_nds_ffb(a, b);
}

static __inline__ long __DEFAULT_FN_ATTRS __riscv_nds_ffzmism(unsigned long a,
                                                              unsigned long b) {
  return __builtin_riscv_nds_ffzmism(a, b);
}

static __inline__ long __DEFAULT_FN_ATTRS __riscv_nds_ffmism(unsigned long a,
                                                             unsigned long b) {
  return __builtin_riscv_nds_ffmism(a, b);
}

static __inline__ long __DEFAULT_FN_ATTRS __riscv_nds_flmism(unsigned long a,
                                                             unsigned long b) {
  return __builtin_riscv_nds_flmism(a, b);
}

#endif // defined(__riscv_nds)

#if defined(__cplusplus)
}
#endif

#endif // define __RISCV_NDS_H
