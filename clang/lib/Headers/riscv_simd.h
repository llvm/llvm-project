/*===---- riscv_simd.h - RISC-V 'Packed SIMD' intrinsics --------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */


#ifndef __RISCV_PACKED_SIMD_H
#define __RISCV_PACKED_SIMD_H

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined (__riscv_p)

#if __riscv_xlen == 32
static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pslli_b(uint32_t __x, int __y) {
  return __builtin_riscv_pslli_b_32(__x, __y);
}

static __inline__ uint32_t __attribute__((__always_inline__, __nodebug__))
__riscv_pslli_h(uint32_t __x, int __y) {
  return __builtin_riscv_pslli_h_32(__x, __y);
}


static __inline__ int32_t __attribute__((__always_inline__, __nodebug__))
__riscv_sadd(int32_t __x, int32_t __y) {
  return __builtin_riscv_sadd(__x, __y);
}
#endif


#if __riscv_xlen == 64
static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pslli_b(uint64_t __x, int __y) {
  return __builtin_riscv_pslli_b_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pslli_h(uint64_t __x, int __y) {
  return __builtin_riscv_pslli_h_64(__x, __y);
}

static __inline__ uint64_t __attribute__((__always_inline__, __nodebug__))
__riscv_pslli_w(uint64_t __x, int __y) {
  return __builtin_riscv_pslli_w(__x, __y);
}
#endif

#endif // defined(__riscv_p)


#if defined(__cplusplus)
}
#endif

#endif
