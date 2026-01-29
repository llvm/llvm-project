/*===---- riscv_simd.h - RISC-V P intrinsics -----------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __RISCV_SIMD_H
#define __RISCV_SIMD_H

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

/* Packed Addition and Subtraction (32-bit) */

static __inline__ int8x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_i8x4(int8x4_t __rs1, int8x4_t __rs2) {
  return __rs1 + __rs2;
}

static __inline__ uint8x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_u8x4(uint8x4_t __rs1, uint8x4_t __rs2) {
  return __rs1 + __rs2;
}

static __inline__ int16x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_i16x2(int16x2_t __rs1, int16x2_t __rs2) {
  return __rs1 + __rs2;
}

static __inline__ uint16x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_u16x2(uint16x2_t __rs1, uint16x2_t __rs2) {
  return __rs1 + __rs2;
}

static __inline__ int8x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_i8x4(int8x4_t __rs1, int8x4_t __rs2) {
  return __rs1 - __rs2;
}

static __inline__ uint8x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_u8x4(uint8x4_t __rs1, uint8x4_t __rs2) {
  return __rs1 - __rs2;
}

static __inline__ int16x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_i16x2(int16x2_t __rs1, int16x2_t __rs2) {
  return __rs1 - __rs2;
}

static __inline__ uint16x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_u16x2(uint16x2_t __rs1, uint16x2_t __rs2) {
  return __rs1 - __rs2;
}

/* Packed Addition and Subtraction (64-bit) */

static __inline__ int8x8_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_i8x8(int8x8_t __rs1, int8x8_t __rs2) {
  return __rs1 + __rs2;
}

static __inline__ uint8x8_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_u8x8(uint8x8_t __rs1, uint8x8_t __rs2) {
  return __rs1 + __rs2;
}

static __inline__ int16x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_i16x4(int16x4_t __rs1, int16x4_t __rs2) {
  return __rs1 + __rs2;
}

static __inline__ uint16x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_u16x4(uint16x4_t __rs1, uint16x4_t __rs2) {
  return __rs1 + __rs2;
}

static __inline__ int32x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_i32x2(int32x2_t __rs1, int32x2_t __rs2) {
  return __rs1 + __rs2;
}

static __inline__ uint32x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_padd_u32x2(uint32x2_t __rs1, uint32x2_t __rs2) {
  return __rs1 + __rs2;
}

static __inline__ int8x8_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_i8x8(int8x8_t __rs1, int8x8_t __rs2) {
  return __rs1 - __rs2;
}

static __inline__ uint8x8_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_u8x8(uint8x8_t __rs1, uint8x8_t __rs2) {
  return __rs1 - __rs2;
}

static __inline__ int16x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_i16x4(int16x4_t __rs1, int16x4_t __rs2) {
  return __rs1 - __rs2;
}

static __inline__ uint16x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_u16x4(uint16x4_t __rs1, uint16x4_t __rs2) {
  return __rs1 - __rs2;
}

static __inline__ int32x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_i32x2(int32x2_t __rs1, int32x2_t __rs2) {
  return __rs1 - __rs2;
}

static __inline__ uint32x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_psub_u32x2(uint32x2_t __rs1, uint32x2_t __rs2) {
  return __rs1 - __rs2;
}

/* Packed Shifts (32-bit) */

static __inline__ uint8x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_s_u8x4(uint8x4_t __rs1, unsigned __shamt) {
  return __rs1 << __shamt;
}

static __inline__ int8x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_s_i8x4(int8x4_t __rs1, unsigned __shamt) {
  return __rs1 << __shamt;
}

static __inline__ uint16x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_s_u16x2(uint16x2_t __rs1, unsigned __shamt) {
  return __rs1 << __shamt;
}

static __inline__ int16x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_s_i16x2(int16x2_t __rs1, unsigned __shamt) {
  return __rs1 << __shamt;
}

static __inline__ uint8x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrl_s_u8x4(uint8x4_t __rs1, unsigned __shamt) {
  return __rs1 >> __shamt;
}

static __inline__ uint16x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrl_s_u16x2(uint16x2_t __rs1, unsigned __shamt) {
  return __rs1 >> __shamt;
}

static __inline__ int8x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_psra_s_i8x4(int8x4_t __rs1, unsigned __shamt) {
  return __rs1 >> __shamt;
}

static __inline__ int16x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_psra_s_i16x2(int16x2_t __rs1, unsigned __shamt) {
  return __rs1 >> __shamt;
}

/* Packed Shifts (64-bit) */

static __inline__ uint8x8_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_s_u8x8(uint8x8_t __rs1, unsigned __shamt) {
  return __rs1 << __shamt;
}

static __inline__ int8x8_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_s_i8x8(int8x8_t __rs1, unsigned __shamt) {
  return __rs1 << __shamt;
}

static __inline__ uint16x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_s_u16x4(uint16x4_t __rs1, unsigned __shamt) {
  return __rs1 << __shamt;
}

static __inline__ int16x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_s_i16x4(int16x4_t __rs1, unsigned __shamt) {
  return __rs1 << __shamt;
}

static __inline__ uint32x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_s_u32x2(uint32x2_t __rs1, unsigned __shamt) {
  return __rs1 << __shamt;
}

static __inline__ int32x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_psll_s_i32x2(int32x2_t __rs1, unsigned __shamt) {
  return __rs1 << __shamt;
}

static __inline__ uint8x8_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrl_s_u8x8(uint8x8_t __rs1, unsigned __shamt) {
  return __rs1 >> __shamt;
}

static __inline__ uint16x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrl_s_u16x4(uint16x4_t __rs1, unsigned __shamt) {
  return __rs1 >> __shamt;
}

static __inline__ uint32x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_psrl_s_u32x2(uint32x2_t __rs1, unsigned __shamt) {
  return __rs1 >> __shamt;
}

static __inline__ int8x8_t __attribute__((__always_inline__, __nodebug__))
__riscv_psra_s_i8x8(int8x8_t __rs1, unsigned __shamt) {
  return __rs1 >> __shamt;
}

static __inline__ int16x4_t __attribute__((__always_inline__, __nodebug__))
__riscv_psra_s_i16x4(int16x4_t __rs1, unsigned __shamt) {
  return __rs1 >> __shamt;
}

static __inline__ int32x2_t __attribute__((__always_inline__, __nodebug__))
__riscv_psra_s_i32x2(int32x2_t __rs1, unsigned __shamt) {
  return __rs1 >> __shamt;
}

#if defined(__cplusplus)
}
#endif

#endif /* __RISCV_SIMD_H */
