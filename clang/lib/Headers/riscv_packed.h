/*===---- riscv_packed.h - RISC-V P intrinsics -----------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __RISCV_PACKED_H
#define __RISCV_PACKED_H

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* Packed SIMD Types */

typedef int8_t int8x4_t __attribute__((vector_size(4)));
typedef uint8_t uint8x4_t __attribute__((vector_size(4)));
typedef int16_t int16x2_t __attribute__((vector_size(4)));
typedef uint16_t uint16x2_t __attribute__((vector_size(4)));

typedef int8_t int8x8_t __attribute__((vector_size(8)));
typedef uint8_t uint8x8_t __attribute__((vector_size(8)));
typedef int16_t int16x4_t __attribute__((vector_size(8)));
typedef uint16_t uint16x4_t __attribute__((vector_size(8)));
typedef int32_t int32x2_t __attribute__((vector_size(8)));
typedef uint32_t uint32x2_t __attribute__((vector_size(8)));

#define _packed_binop(name, retty, ty1, ty2, op)                               \
  static __inline__ retty __attribute__((__always_inline__, __nodebug__))      \
  __riscv_##name(ty1 __rs1, ty2 __rs2) {                                       \
    return __rs1 op __rs2;                                                     \
  }

#define _packed_addsub(name, ty, op) _packed_binop(name, ty, ty, ty, op)
#define _packed_shift(name, ty, op) _packed_binop(name, ty, ty, unsigned, op)

/* Packed Addition and Subtraction (32-bit) */
_packed_addsub(padd_i8x4, int8x4_t, +)
_packed_addsub(padd_u8x4, uint8x4_t, +)
_packed_addsub(padd_i16x2, int16x2_t, +)
_packed_addsub(padd_u16x2, uint16x2_t, +)
_packed_addsub(psub_i8x4, int8x4_t, -)
_packed_addsub(psub_u8x4, uint8x4_t, -)
_packed_addsub(psub_i16x2, int16x2_t, -)
_packed_addsub(psub_u16x2, uint16x2_t, -)

/* Packed Addition and Subtraction (64-bit) */
_packed_addsub(padd_i8x8, int8x8_t, +)
_packed_addsub(padd_u8x8, uint8x8_t, +)
_packed_addsub(padd_i16x4, int16x4_t, +)
_packed_addsub(padd_u16x4, uint16x4_t, +)
_packed_addsub(padd_i32x2, int32x2_t, +)
_packed_addsub(padd_u32x2, uint32x2_t, +)
_packed_addsub(psub_i8x8, int8x8_t, -)
_packed_addsub(psub_u8x8, uint8x8_t, -)
_packed_addsub(psub_i16x4, int16x4_t, -)
_packed_addsub(psub_u16x4, uint16x4_t, -)
_packed_addsub(psub_i32x2, int32x2_t, -)
_packed_addsub(psub_u32x2, uint32x2_t, -)

/* Packed Shifts (32-bit) */
_packed_shift(psll_s_u8x4, uint8x4_t, <<)
_packed_shift(psll_s_i8x4, int8x4_t, <<)
_packed_shift(psll_s_u16x2, uint16x2_t, <<)
_packed_shift(psll_s_i16x2, int16x2_t, <<)
_packed_shift(psrl_s_u8x4, uint8x4_t, >>)
_packed_shift(psrl_s_u16x2, uint16x2_t, >>)
_packed_shift(psra_s_i8x4, int8x4_t, >>)
_packed_shift(psra_s_i16x2, int16x2_t, >>)

/* Packed Shifts (64-bit) */
_packed_shift(psll_s_u8x8, uint8x8_t, <<)
_packed_shift(psll_s_i8x8, int8x8_t, <<)
_packed_shift(psll_s_u16x4, uint16x4_t, <<)
_packed_shift(psll_s_i16x4, int16x4_t, <<)
_packed_shift(psll_s_u32x2, uint32x2_t, <<)
_packed_shift(psll_s_i32x2, int32x2_t, <<)
_packed_shift(psrl_s_u8x8, uint8x8_t, >>)
_packed_shift(psrl_s_u16x4, uint16x4_t, >>)
_packed_shift(psrl_s_u32x2, uint32x2_t, >>)
_packed_shift(psra_s_i8x8, int8x8_t, >>)
_packed_shift(psra_s_i16x4, int16x4_t, >>)
_packed_shift(psra_s_i32x2, int32x2_t, >>)

#undef _packed_addsub
#undef _packed_shift
#undef _packed_binop

#if defined(__cplusplus)
}
#endif

#endif /* __RISCV_PACKED_H */
