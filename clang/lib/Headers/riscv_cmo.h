/*===---- riscv_cmo.h - RISC-V CMO intrinsics ----------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __RISCV_CMO_H
#define __RISCV_CMO_H

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__riscv_zicboz)
static __inline__ void __attribute__((__always_inline__, __nodebug__))
__riscv_cbo_zero(void *__x) {
  return __builtin_riscv_cbo_zero(__x);
}
#endif // defined(__riscv_zicboz)

#if defined(__cplusplus)
}
#endif

#endif // define __RISCV_CMO_H
