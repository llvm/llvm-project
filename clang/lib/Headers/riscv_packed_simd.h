/*===---- riscv_packed_simd.h - RISC-V P intrinsics ------------------------===
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

/* Packed SIMD Types */

typedef int8_t int8x4_t __attribute__((__vector_size__(4)));
typedef uint8_t uint8x4_t __attribute__((__vector_size__(4)));
typedef int16_t int16x2_t __attribute__((__vector_size__(4)));
typedef uint16_t uint16x2_t __attribute__((__vector_size__(4)));

typedef int8_t int8x8_t __attribute__((__vector_size__(8)));
typedef uint8_t uint8x8_t __attribute__((__vector_size__(8)));
typedef int16_t int16x4_t __attribute__((__vector_size__(8)));
typedef uint16_t uint16x4_t __attribute__((__vector_size__(8)));
typedef int32_t int32x2_t __attribute__((__vector_size__(8)));
typedef uint32_t uint32x2_t __attribute__((__vector_size__(8)));

#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__))

#define __packed_splat2(ty, x) ((ty){(x), (x)})
#define __packed_splat4(ty, x) ((ty){(x), (x), (x), (x)})
#define __packed_splat8(ty, x) ((ty){(x), (x), (x), (x), (x), (x), (x), (x)})

#define __packed_splat(name, ty, scalar_ty, splat)                             \
  static __inline__ ty __DEFAULT_FN_ATTRS __riscv_##name(scalar_ty __x) {      \
    return splat(ty, __x);                                                     \
  }

#define __packed_shift(name, ty, op, mask)                                     \
  static __inline__ ty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1,             \
                                                         unsigned __rs2) {     \
    return __rs1 op(__rs2 & (mask));                                           \
  }
#define __packed_shift8(name, ty, op) __packed_shift(name, ty, op, 0x7)
#define __packed_shift16(name, ty, op) __packed_shift(name, ty, op, 0xf)
#define __packed_shift32(name, ty, op) __packed_shift(name, ty, op, 0x1f)

#define __packed_scalar_binary_op(name, ty, scalar_ty, op, splat)              \
  static __inline__ ty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1,             \
                                                         scalar_ty __rs2) {    \
    return __rs1 op splat(ty, __rs2);                                          \
  }

#define __packed_binary_op(name, ty, op)                                       \
  static __inline__ ty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1, ty __rs2) { \
    return __rs1 op __rs2;                                                     \
  }

#define __packed_unary_op(name, ty, op)                                        \
  static __inline__ ty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1) {           \
    return op __rs1;                                                           \
  }

#define __packed_binary_builtin(name, ty, builtin)                             \
  static __inline__ ty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1, ty __rs2) { \
    return builtin(__rs1, __rs2);                                              \
  }

#define __packed_sh1add(name, ty)                                              \
  static __inline__ ty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1, ty __rs2) { \
    return (__rs1 << 1) + __rs2;                                               \
  }

/* TODO: switch to sadd_sat(__builtin_elementwise_shl_sat(a, 1), b) once a
 * generic elementwise shl_sat builtin exists. sadd_sat(a, a) is equivalent
 * for signed types and the backend's saturating_shl1 PatFrags matches both
 * shapes. */
#define __packed_sh1sadd(name, ty)                                             \
  static __inline__ ty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1, ty __rs2) { \
    return __builtin_elementwise_add_sat(                                      \
        __builtin_elementwise_add_sat(__rs1, __rs1), __rs2);                   \
  }

#define __packed_cmp(name, ty, rty, op)                                        \
  static __inline__ rty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1,            \
                                                          ty __rs2) {          \
    return (rty)(__rs1 op __rs2);                                              \
  }

#define __packed_pabs(name, ty, rty)                                           \
  static __inline__ rty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1) {          \
    return (rty)__builtin_elementwise_abs(__rs1);                              \
  }

#define __packed_binary_builtin_cast(name, ty, rty, builtin)                   \
  static __inline__ rty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1,            \
                                                          ty __rs2) {          \
    return (rty)builtin(__rs1, __rs2);                                         \
  }

#define __packed_reduction(name, rty, ty, builtin)                             \
  static __inline__ rty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1,            \
                                                          rty __rs2) {         \
    return builtin(__rs1, __rs2);                                              \
  }

#define __packed_merge_builtin(name, ty, mask_ty, builtin)                     \
  static __inline__ ty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1, ty __rs2,   \
                                                         mask_ty __rd) {       \
    return (ty)builtin(__rs1, __rs2, __rd);                                    \
  }

#define __packed_psabs(name, ty, builtin)                                      \
  static __inline__ ty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1) {           \
    return builtin(__rs1);                                                     \
  }

/* Packed Reverse: reverse the order of the elements. Lowered to a single
 * rev8/rev16/ppairoe.* by the backend's packed reverse-shuffle handling. */
#define __packed_reverse2(name, ty)                                            \
  static __inline__ ty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1) {           \
    return __builtin_shufflevector(__rs1, __rs1, 1, 0);                        \
  }
#define __packed_reverse4(name, ty)                                            \
  static __inline__ ty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1) {           \
    return __builtin_shufflevector(__rs1, __rs1, 3, 2, 1, 0);                  \
  }
#define __packed_reverse8(name, ty)                                            \
  static __inline__ ty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1) {           \
    return __builtin_shufflevector(__rs1, __rs1, 7, 6, 5, 4, 3, 2, 1, 0);      \
  }

#define __packed_zip2(name, rty, ty)                                           \
  static __inline__ rty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1,            \
                                                          ty __rs2) {          \
    return __builtin_shufflevector(__rs1, __rs2, 0, 2, 1, 3);                  \
  }
#define __packed_zip4(name, rty, ty)                                           \
  static __inline__ rty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1,            \
                                                          ty __rs2) {          \
    return __builtin_shufflevector(__rs1, __rs2, 0, 4, 1, 5, 2, 6, 3, 7);      \
  }
#define __packed_unzipe2(name, rty, ty)                                        \
  static __inline__ rty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1) {          \
    return __builtin_shufflevector(__rs1, __rs1, 0, 2);                        \
  }
#define __packed_unzipe4(name, rty, ty)                                        \
  static __inline__ rty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1) {          \
    return __builtin_shufflevector(__rs1, __rs1, 0, 2, 4, 6);                  \
  }
#define __packed_unzipo2(name, rty, ty)                                        \
  static __inline__ rty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1) {          \
    return __builtin_shufflevector(__rs1, __rs1, 1, 3);                        \
  }
#define __packed_unzipo4(name, rty, ty)                                        \
  static __inline__ rty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1) {          \
    return __builtin_shufflevector(__rs1, __rs1, 1, 3, 5, 7);                  \
  }

#define __packed_abdsum(name, rty, ty, builtin)                                \
  static __inline__ rty __DEFAULT_FN_ATTRS __riscv_##name(ty __rs1,            \
                                                          ty __rs2) {          \
    return builtin(__rs1, __rs2);                                              \
  }

#define __packed_abdsum_acc(name, rty, ty, builtin)                            \
  static __inline__ rty __DEFAULT_FN_ATTRS __riscv_##name(rty __rd, ty __rs1,  \
                                                          ty __rs2) {          \
    return builtin(__rd, __rs1, __rs2);                                        \
  }

// clang-format off: macro call sites have no trailing semicolons, which
// confuses clang-format into a deeply nested expression.

/* Packed Splat (32-bit) */
__packed_splat(pmv_s_u8x4, uint8x4_t, uint8_t, __packed_splat4)
__packed_splat(pmv_s_i8x4, int8x4_t, int8_t, __packed_splat4)
__packed_splat(pmv_s_u16x2, uint16x2_t, uint16_t, __packed_splat2)
__packed_splat(pmv_s_i16x2, int16x2_t, int16_t, __packed_splat2)

/* Packed Splat (64-bit) */
__packed_splat(pmv_s_u8x8, uint8x8_t, uint8_t, __packed_splat8)
__packed_splat(pmv_s_i8x8, int8x8_t, int8_t, __packed_splat8)
__packed_splat(pmv_s_u16x4, uint16x4_t, uint16_t, __packed_splat4)
__packed_splat(pmv_s_i16x4, int16x4_t, int16_t, __packed_splat4)
__packed_splat(pmv_s_u32x2, uint32x2_t, uint32_t, __packed_splat2)
__packed_splat(pmv_s_i32x2, int32x2_t, int32_t, __packed_splat2)

/* Packed Addition and Subtraction (32-bit) */
__packed_binary_op(padd_i8x4, int8x4_t, +)
__packed_binary_op(padd_u8x4, uint8x4_t, +)
__packed_binary_op(padd_i16x2, int16x2_t, +)
__packed_binary_op(padd_u16x2, uint16x2_t, +)
__packed_binary_op(psub_i8x4, int8x4_t, -)
__packed_binary_op(psub_u8x4, uint8x4_t, -)
__packed_binary_op(psub_i16x2, int16x2_t, -)
__packed_binary_op(psub_u16x2, uint16x2_t, -)
__packed_unary_op(pneg_i8x4, int8x4_t, -)
__packed_unary_op(pneg_i16x2, int16x2_t, -)

/* Packed Addition and Subtraction (64-bit) */
__packed_binary_op(padd_i8x8, int8x8_t, +)
__packed_binary_op(padd_u8x8, uint8x8_t, +)
__packed_binary_op(padd_i16x4, int16x4_t, +)
__packed_binary_op(padd_u16x4, uint16x4_t, +)
__packed_binary_op(padd_i32x2, int32x2_t, +)
__packed_binary_op(padd_u32x2, uint32x2_t, +)
__packed_binary_op(psub_i8x8, int8x8_t, -)
__packed_binary_op(psub_u8x8, uint8x8_t, -)
__packed_binary_op(psub_i16x4, int16x4_t, -)
__packed_binary_op(psub_u16x4, uint16x4_t, -)
__packed_binary_op(psub_i32x2, int32x2_t, -)
__packed_binary_op(psub_u32x2, uint32x2_t, -)
__packed_unary_op(pneg_i8x8, int8x8_t, -)
__packed_unary_op(pneg_i16x4, int16x4_t, -)
__packed_unary_op(pneg_i32x2, int32x2_t, -)

/* Packed Addition with Scalar (32-bit) */
__packed_scalar_binary_op(padd_s_u8x4, uint8x4_t, uint8_t, +, __packed_splat4)
__packed_scalar_binary_op(padd_s_i8x4, int8x4_t, int8_t, +, __packed_splat4)
__packed_scalar_binary_op(padd_s_u16x2, uint16x2_t, uint16_t, +,
                          __packed_splat2)
__packed_scalar_binary_op(padd_s_i16x2, int16x2_t, int16_t, +,
                          __packed_splat2)

/* Packed Addition with Scalar (64-bit) */
__packed_scalar_binary_op(padd_s_u8x8, uint8x8_t, uint8_t, +, __packed_splat8)
__packed_scalar_binary_op(padd_s_i8x8, int8x8_t, int8_t, +, __packed_splat8)
__packed_scalar_binary_op(padd_s_u16x4, uint16x4_t, uint16_t, +,
                          __packed_splat4)
__packed_scalar_binary_op(padd_s_i16x4, int16x4_t, int16_t, +,
                          __packed_splat4)
__packed_scalar_binary_op(padd_s_u32x2, uint32x2_t, uint32_t, +,
                          __packed_splat2)
__packed_scalar_binary_op(padd_s_i32x2, int32x2_t, int32_t, +,
                          __packed_splat2)

/* Packed Saturating Addition and Subtraction (32-bit) */
__packed_binary_builtin(psadd_i8x4, int8x4_t, __builtin_elementwise_add_sat)
__packed_binary_builtin(psadd_i16x2, int16x2_t, __builtin_elementwise_add_sat)
__packed_binary_builtin(psaddu_u8x4, uint8x4_t, __builtin_elementwise_add_sat)
__packed_binary_builtin(psaddu_u16x2, uint16x2_t, __builtin_elementwise_add_sat)
__packed_binary_builtin(pssub_i8x4, int8x4_t, __builtin_elementwise_sub_sat)
__packed_binary_builtin(pssub_i16x2, int16x2_t, __builtin_elementwise_sub_sat)
__packed_binary_builtin(pssubu_u8x4, uint8x4_t, __builtin_elementwise_sub_sat)
__packed_binary_builtin(pssubu_u16x2, uint16x2_t, __builtin_elementwise_sub_sat)

/* Packed Saturating Addition and Subtraction (64-bit) */
__packed_binary_builtin(psadd_i8x8, int8x8_t, __builtin_elementwise_add_sat)
__packed_binary_builtin(psadd_i16x4, int16x4_t, __builtin_elementwise_add_sat)
__packed_binary_builtin(psadd_i32x2, int32x2_t, __builtin_elementwise_add_sat)
__packed_binary_builtin(psaddu_u8x8, uint8x8_t, __builtin_elementwise_add_sat)
__packed_binary_builtin(psaddu_u16x4, uint16x4_t, __builtin_elementwise_add_sat)
__packed_binary_builtin(psaddu_u32x2, uint32x2_t, __builtin_elementwise_add_sat)
__packed_binary_builtin(pssub_i8x8, int8x8_t, __builtin_elementwise_sub_sat)
__packed_binary_builtin(pssub_i16x4, int16x4_t, __builtin_elementwise_sub_sat)
__packed_binary_builtin(pssub_i32x2, int32x2_t, __builtin_elementwise_sub_sat)
__packed_binary_builtin(pssubu_u8x8, uint8x8_t, __builtin_elementwise_sub_sat)
__packed_binary_builtin(pssubu_u16x4, uint16x4_t, __builtin_elementwise_sub_sat)
__packed_binary_builtin(pssubu_u32x2, uint32x2_t, __builtin_elementwise_sub_sat)

/* Packed Shift-Add (32-bit) */
__packed_sh1add(psh1add_i16x2, int16x2_t)
__packed_sh1add(psh1add_u16x2, uint16x2_t)
__packed_sh1sadd(pssh1sadd_i16x2, int16x2_t)

/* Packed Shift-Add (64-bit) */
__packed_sh1add(psh1add_i16x4, int16x4_t)
__packed_sh1add(psh1add_u16x4, uint16x4_t)
__packed_sh1add(psh1add_i32x2, int32x2_t)
__packed_sh1add(psh1add_u32x2, uint32x2_t)
__packed_sh1sadd(pssh1sadd_i16x4, int16x4_t)
__packed_sh1sadd(pssh1sadd_i32x2, int32x2_t)

/* Packed Exchanged Addition and Subtraction (32-bit) */
__packed_binary_builtin(pas_x_i16x2, int16x2_t, __builtin_riscv_pas_x_i16x2)
__packed_binary_builtin(psa_x_i16x2, int16x2_t, __builtin_riscv_psa_x_i16x2)
__packed_binary_builtin(psas_x_i16x2, int16x2_t, __builtin_riscv_psas_x_i16x2)
__packed_binary_builtin(pssa_x_i16x2, int16x2_t, __builtin_riscv_pssa_x_i16x2)
__packed_binary_builtin(paas_x_i16x2, int16x2_t, __builtin_riscv_paas_x_i16x2)
__packed_binary_builtin(pasa_x_i16x2, int16x2_t, __builtin_riscv_pasa_x_i16x2)

/* Packed Exchanged Addition and Subtraction (64-bit) */
__packed_binary_builtin(pas_x_i16x4, int16x4_t, __builtin_riscv_pas_x_i16x4)
__packed_binary_builtin(psa_x_i16x4, int16x4_t, __builtin_riscv_psa_x_i16x4)
__packed_binary_builtin(psas_x_i16x4, int16x4_t, __builtin_riscv_psas_x_i16x4)
__packed_binary_builtin(pssa_x_i16x4, int16x4_t, __builtin_riscv_pssa_x_i16x4)
__packed_binary_builtin(paas_x_i16x4, int16x4_t, __builtin_riscv_paas_x_i16x4)
__packed_binary_builtin(pasa_x_i16x4, int16x4_t, __builtin_riscv_pasa_x_i16x4)
__packed_binary_builtin(pas_x_i32x2, int32x2_t, __builtin_riscv_pas_x_i32x2)
__packed_binary_builtin(psa_x_i32x2, int32x2_t, __builtin_riscv_psa_x_i32x2)
__packed_binary_builtin(psas_x_i32x2, int32x2_t, __builtin_riscv_psas_x_i32x2)
__packed_binary_builtin(pssa_x_i32x2, int32x2_t, __builtin_riscv_pssa_x_i32x2)
__packed_binary_builtin(paas_x_i32x2, int32x2_t, __builtin_riscv_paas_x_i32x2)
__packed_binary_builtin(pasa_x_i32x2, int32x2_t, __builtin_riscv_pasa_x_i32x2)

/* Packed Minimum and Maximum (32-bit) */
__packed_binary_builtin(pmin_i8x4, int8x4_t, __builtin_elementwise_min)
__packed_binary_builtin(pmin_i16x2, int16x2_t, __builtin_elementwise_min)
__packed_binary_builtin(pminu_u8x4, uint8x4_t, __builtin_elementwise_min)
__packed_binary_builtin(pminu_u16x2, uint16x2_t, __builtin_elementwise_min)
__packed_binary_builtin(pmax_i8x4, int8x4_t, __builtin_elementwise_max)
__packed_binary_builtin(pmax_i16x2, int16x2_t, __builtin_elementwise_max)
__packed_binary_builtin(pmaxu_u8x4, uint8x4_t, __builtin_elementwise_max)
__packed_binary_builtin(pmaxu_u16x2, uint16x2_t, __builtin_elementwise_max)

/* Packed Minimum and Maximum (64-bit) */
__packed_binary_builtin(pmin_i8x8, int8x8_t, __builtin_elementwise_min)
__packed_binary_builtin(pmin_i16x4, int16x4_t, __builtin_elementwise_min)
__packed_binary_builtin(pmin_i32x2, int32x2_t, __builtin_elementwise_min)
__packed_binary_builtin(pminu_u8x8, uint8x8_t, __builtin_elementwise_min)
__packed_binary_builtin(pminu_u16x4, uint16x4_t, __builtin_elementwise_min)
__packed_binary_builtin(pminu_u32x2, uint32x2_t, __builtin_elementwise_min)
__packed_binary_builtin(pmax_i8x8, int8x8_t, __builtin_elementwise_max)
__packed_binary_builtin(pmax_i16x4, int16x4_t, __builtin_elementwise_max)
__packed_binary_builtin(pmax_i32x2, int32x2_t, __builtin_elementwise_max)
__packed_binary_builtin(pmaxu_u8x8, uint8x8_t, __builtin_elementwise_max)
__packed_binary_builtin(pmaxu_u16x4, uint16x4_t, __builtin_elementwise_max)
__packed_binary_builtin(pmaxu_u32x2, uint32x2_t, __builtin_elementwise_max)

/* Packed Comparison (32-bit) */
__packed_cmp(pmseq_i8x4_u8x4, int8x4_t, uint8x4_t, ==)
__packed_cmp(pmseq_u8x4_u8x4, uint8x4_t, uint8x4_t, ==)
__packed_cmp(pmsne_i8x4_u8x4, int8x4_t, uint8x4_t, !=)
__packed_cmp(pmsne_u8x4_u8x4, uint8x4_t, uint8x4_t, !=)
__packed_cmp(pmslt_u8x4, int8x4_t, uint8x4_t, <)
__packed_cmp(pmsltu_u8x4, uint8x4_t, uint8x4_t, <)
__packed_cmp(pmsgt_u8x4, int8x4_t, uint8x4_t, >)
__packed_cmp(pmsgtu_u8x4, uint8x4_t, uint8x4_t, >)
__packed_cmp(pmsge_u8x4, int8x4_t, uint8x4_t, >=)
__packed_cmp(pmsgeu_u8x4, uint8x4_t, uint8x4_t, >=)
__packed_cmp(pmsle_u8x4, int8x4_t, uint8x4_t, <=)
__packed_cmp(pmsleu_u8x4, uint8x4_t, uint8x4_t, <=)
__packed_cmp(pmseq_i16x2_u16x2, int16x2_t, uint16x2_t, ==)
__packed_cmp(pmseq_u16x2_u16x2, uint16x2_t, uint16x2_t, ==)
__packed_cmp(pmsne_i16x2_u16x2, int16x2_t, uint16x2_t, !=)
__packed_cmp(pmsne_u16x2_u16x2, uint16x2_t, uint16x2_t, !=)
__packed_cmp(pmslt_u16x2, int16x2_t, uint16x2_t, <)
__packed_cmp(pmsltu_u16x2, uint16x2_t, uint16x2_t, <)
__packed_cmp(pmsgt_u16x2, int16x2_t, uint16x2_t, >)
__packed_cmp(pmsgtu_u16x2, uint16x2_t, uint16x2_t, >)
__packed_cmp(pmsge_u16x2, int16x2_t, uint16x2_t, >=)
__packed_cmp(pmsgeu_u16x2, uint16x2_t, uint16x2_t, >=)
__packed_cmp(pmsle_u16x2, int16x2_t, uint16x2_t, <=)
__packed_cmp(pmsleu_u16x2, uint16x2_t, uint16x2_t, <=)

/* Packed Comparison (64-bit) */
__packed_cmp(pmseq_i8x8_u8x8, int8x8_t, uint8x8_t, ==)
__packed_cmp(pmseq_u8x8_u8x8, uint8x8_t, uint8x8_t, ==)
__packed_cmp(pmsne_i8x8_u8x8, int8x8_t, uint8x8_t, !=)
__packed_cmp(pmsne_u8x8_u8x8, uint8x8_t, uint8x8_t, !=)
__packed_cmp(pmslt_u8x8, int8x8_t, uint8x8_t, <)
__packed_cmp(pmsltu_u8x8, uint8x8_t, uint8x8_t, <)
__packed_cmp(pmsgt_u8x8, int8x8_t, uint8x8_t, >)
__packed_cmp(pmsgtu_u8x8, uint8x8_t, uint8x8_t, >)
__packed_cmp(pmsge_u8x8, int8x8_t, uint8x8_t, >=)
__packed_cmp(pmsgeu_u8x8, uint8x8_t, uint8x8_t, >=)
__packed_cmp(pmsle_u8x8, int8x8_t, uint8x8_t, <=)
__packed_cmp(pmsleu_u8x8, uint8x8_t, uint8x8_t, <=)
__packed_cmp(pmseq_i16x4_u16x4, int16x4_t, uint16x4_t, ==)
__packed_cmp(pmseq_u16x4_u16x4, uint16x4_t, uint16x4_t, ==)
__packed_cmp(pmsne_i16x4_u16x4, int16x4_t, uint16x4_t, !=)
__packed_cmp(pmsne_u16x4_u16x4, uint16x4_t, uint16x4_t, !=)
__packed_cmp(pmslt_u16x4, int16x4_t, uint16x4_t, <)
__packed_cmp(pmsltu_u16x4, uint16x4_t, uint16x4_t, <)
__packed_cmp(pmsgt_u16x4, int16x4_t, uint16x4_t, >)
__packed_cmp(pmsgtu_u16x4, uint16x4_t, uint16x4_t, >)
__packed_cmp(pmsge_u16x4, int16x4_t, uint16x4_t, >=)
__packed_cmp(pmsgeu_u16x4, uint16x4_t, uint16x4_t, >=)
__packed_cmp(pmsle_u16x4, int16x4_t, uint16x4_t, <=)
__packed_cmp(pmsleu_u16x4, uint16x4_t, uint16x4_t, <=)
__packed_cmp(pmseq_i32x2_u32x2, int32x2_t, uint32x2_t, ==)
__packed_cmp(pmseq_u32x2_u32x2, uint32x2_t, uint32x2_t, ==)
__packed_cmp(pmsne_i32x2_u32x2, int32x2_t, uint32x2_t, !=)
__packed_cmp(pmsne_u32x2_u32x2, uint32x2_t, uint32x2_t, !=)
__packed_cmp(pmslt_u32x2, int32x2_t, uint32x2_t, <)
__packed_cmp(pmsltu_u32x2, uint32x2_t, uint32x2_t, <)
__packed_cmp(pmsgt_u32x2, int32x2_t, uint32x2_t, >)
__packed_cmp(pmsgtu_u32x2, uint32x2_t, uint32x2_t, >)
__packed_cmp(pmsge_u32x2, int32x2_t, uint32x2_t, >=)
__packed_cmp(pmsgeu_u32x2, uint32x2_t, uint32x2_t, >=)
__packed_cmp(pmsle_u32x2, int32x2_t, uint32x2_t, <=)
__packed_cmp(pmsleu_u32x2, uint32x2_t, uint32x2_t, <=)

/* Packed Shifts (32-bit) */
__packed_shift8(psll_s_u8x4, uint8x4_t, <<)
__packed_shift8(psll_s_i8x4, int8x4_t, <<)
__packed_shift16(psll_s_u16x2, uint16x2_t, <<)
__packed_shift16(psll_s_i16x2, int16x2_t, <<)
__packed_shift8(psrl_s_u8x4, uint8x4_t, >>)
__packed_shift16(psrl_s_u16x2, uint16x2_t, >>)
__packed_shift8(psra_s_i8x4, int8x4_t, >>)
__packed_shift16(psra_s_i16x2, int16x2_t, >>)

/* Packed Shifts (64-bit) */
__packed_shift8(psll_s_u8x8, uint8x8_t, <<)
__packed_shift8(psll_s_i8x8, int8x8_t, <<)
__packed_shift16(psll_s_u16x4, uint16x4_t, <<)
__packed_shift16(psll_s_i16x4, int16x4_t, <<)
__packed_shift32(psll_s_u32x2, uint32x2_t, <<)
__packed_shift32(psll_s_i32x2, int32x2_t, <<)
__packed_shift8(psrl_s_u8x8, uint8x8_t, >>)
__packed_shift16(psrl_s_u16x4, uint16x4_t, >>)
__packed_shift32(psrl_s_u32x2, uint32x2_t, >>)
__packed_shift8(psra_s_i8x8, int8x8_t, >>)
__packed_shift16(psra_s_i16x4, int16x4_t, >>)
__packed_shift32(psra_s_i32x2, int32x2_t, >>)

/* Packed Logical Operations (32-bit) */
__packed_binary_op(pand_i8x4, int8x4_t, &)
__packed_binary_op(pand_u8x4, uint8x4_t, &)
__packed_binary_op(pand_i16x2, int16x2_t, &)
__packed_binary_op(pand_u16x2, uint16x2_t, &)
__packed_binary_op(por_i8x4, int8x4_t, |)
__packed_binary_op(por_u8x4, uint8x4_t, |)
__packed_binary_op(por_i16x2, int16x2_t, |)
__packed_binary_op(por_u16x2, uint16x2_t, |)
__packed_binary_op(pxor_i8x4, int8x4_t, ^)
__packed_binary_op(pxor_u8x4, uint8x4_t, ^)
__packed_binary_op(pxor_i16x2, int16x2_t, ^)
__packed_binary_op(pxor_u16x2, uint16x2_t, ^)
__packed_unary_op(pnot_i8x4, int8x4_t, ~)
__packed_unary_op(pnot_u8x4, uint8x4_t, ~)
__packed_unary_op(pnot_i16x2, int16x2_t, ~)
__packed_unary_op(pnot_u16x2, uint16x2_t, ~)

/* Packed Logical Operations (64-bit) */
__packed_binary_op(pand_i8x8, int8x8_t, &)
__packed_binary_op(pand_u8x8, uint8x8_t, &)
__packed_binary_op(pand_i16x4, int16x4_t, &)
__packed_binary_op(pand_u16x4, uint16x4_t, &)
__packed_binary_op(pand_i32x2, int32x2_t, &)
__packed_binary_op(pand_u32x2, uint32x2_t, &)
__packed_binary_op(por_i8x8, int8x8_t, |)
__packed_binary_op(por_u8x8, uint8x8_t, |)
__packed_binary_op(por_i16x4, int16x4_t, |)
__packed_binary_op(por_u16x4, uint16x4_t, |)
__packed_binary_op(por_i32x2, int32x2_t, |)
__packed_binary_op(por_u32x2, uint32x2_t, |)
__packed_binary_op(pxor_i8x8, int8x8_t, ^)
__packed_binary_op(pxor_u8x8, uint8x8_t, ^)
__packed_binary_op(pxor_i16x4, int16x4_t, ^)
__packed_binary_op(pxor_u16x4, uint16x4_t, ^)
__packed_binary_op(pxor_i32x2, int32x2_t, ^)
__packed_binary_op(pxor_u32x2, uint32x2_t, ^)
__packed_unary_op(pnot_i8x8, int8x8_t, ~)
__packed_unary_op(pnot_u8x8, uint8x8_t, ~)
__packed_unary_op(pnot_i16x4, int16x4_t, ~)
__packed_unary_op(pnot_u16x4, uint16x4_t, ~)
__packed_unary_op(pnot_i32x2, int32x2_t, ~)
__packed_unary_op(pnot_u32x2, uint32x2_t, ~)

/* Packed Reverse (32-bit) */
__packed_reverse4(prev_i8x4, int8x4_t)
__packed_reverse4(prev_u8x4, uint8x4_t)
__packed_reverse2(prev_i16x2, int16x2_t)
__packed_reverse2(prev_u16x2, uint16x2_t)

/* Packed Reverse (64-bit) */
__packed_reverse8(prev_i8x8, int8x8_t)
__packed_reverse8(prev_u8x8, uint8x8_t)
__packed_reverse4(prev_i16x4, int16x4_t)
__packed_reverse4(prev_u16x4, uint16x4_t)
__packed_reverse2(prev_i32x2, int32x2_t)
__packed_reverse2(prev_u32x2, uint32x2_t)

/* Packed Zip */
__packed_zip4(pzip_i8x8, int8x8_t, int8x4_t)
__packed_zip4(pzip_u8x8, uint8x8_t, uint8x4_t)
__packed_zip2(pzip_i16x4, int16x4_t, int16x2_t)
__packed_zip2(pzip_u16x4, uint16x4_t, uint16x2_t)

/* Packed Unzip */
__packed_unzipe4(punzipe_i8x4, int8x4_t, int8x8_t)
__packed_unzipo4(punzipo_i8x4, int8x4_t, int8x8_t)
__packed_unzipe4(punzipe_u8x4, uint8x4_t, uint8x8_t)
__packed_unzipo4(punzipo_u8x4, uint8x4_t, uint8x8_t)
__packed_unzipe2(punzipe_i16x2, int16x2_t, int16x4_t)
__packed_unzipo2(punzipo_i16x2, int16x2_t, int16x4_t)
__packed_unzipe2(punzipe_u16x2, uint16x2_t, uint16x4_t)
__packed_unzipo2(punzipo_u16x2, uint16x2_t, uint16x4_t)

/* Packed Averaging Addition and Subtraction (32-bit) */
__packed_binary_builtin(paadd_i8x4, int8x4_t, __builtin_riscv_paadd_i8x4)
__packed_binary_builtin(paadd_i16x2, int16x2_t, __builtin_riscv_paadd_i16x2)
__packed_binary_builtin(paaddu_u8x4, uint8x4_t, __builtin_riscv_paaddu_u8x4)
__packed_binary_builtin(paaddu_u16x2, uint16x2_t, __builtin_riscv_paaddu_u16x2)
__packed_binary_builtin(pasub_i8x4, int8x4_t, __builtin_riscv_pasub_i8x4)
__packed_binary_builtin(pasub_i16x2, int16x2_t, __builtin_riscv_pasub_i16x2)
__packed_binary_builtin(pasubu_u8x4, uint8x4_t, __builtin_riscv_pasubu_u8x4)
__packed_binary_builtin(pasubu_u16x2, uint16x2_t, __builtin_riscv_pasubu_u16x2)

/* Packed Averaging Addition and Subtraction (64-bit) */
__packed_binary_builtin(paadd_i8x8, int8x8_t, __builtin_riscv_paadd_i8x8)
__packed_binary_builtin(paadd_i16x4, int16x4_t, __builtin_riscv_paadd_i16x4)
__packed_binary_builtin(paadd_i32x2, int32x2_t, __builtin_riscv_paadd_i32x2)
__packed_binary_builtin(paaddu_u8x8, uint8x8_t, __builtin_riscv_paaddu_u8x8)
__packed_binary_builtin(paaddu_u16x4, uint16x4_t, __builtin_riscv_paaddu_u16x4)
__packed_binary_builtin(paaddu_u32x2, uint32x2_t, __builtin_riscv_paaddu_u32x2)
__packed_binary_builtin(pasub_i8x8, int8x8_t, __builtin_riscv_pasub_i8x8)
__packed_binary_builtin(pasub_i16x4, int16x4_t, __builtin_riscv_pasub_i16x4)
__packed_binary_builtin(pasub_i32x2, int32x2_t, __builtin_riscv_pasub_i32x2)
__packed_binary_builtin(pasubu_u8x8, uint8x8_t, __builtin_riscv_pasubu_u8x8)
__packed_binary_builtin(pasubu_u16x4, uint16x4_t, __builtin_riscv_pasubu_u16x4)
__packed_binary_builtin(pasubu_u32x2, uint32x2_t, __builtin_riscv_pasubu_u32x2)

/* Packed Absolute Value and Absolute Difference (32-bit) */
__packed_pabs(pabs_i8x4, int8x4_t, uint8x4_t)
__packed_pabs(pabs_i16x2, int16x2_t, uint16x2_t)
__packed_binary_builtin_cast(pabd_i8x4, int8x4_t, uint8x4_t, __builtin_riscv_pabd_i8x4)
__packed_binary_builtin_cast(pabd_i16x2, int16x2_t, uint16x2_t, __builtin_riscv_pabd_i16x2)
__packed_binary_builtin_cast(pabdu_u8x4, uint8x4_t, uint8x4_t, __builtin_riscv_pabdu_u8x4)
__packed_binary_builtin_cast(pabdu_u16x2, uint16x2_t, uint16x2_t, __builtin_riscv_pabdu_u16x2)

/* Packed Absolute Value and Absolute Difference (64-bit) */
__packed_pabs(pabs_i8x8, int8x8_t, uint8x8_t)
__packed_pabs(pabs_i16x4, int16x4_t, uint16x4_t)
__packed_binary_builtin_cast(pabd_i8x8, int8x8_t, uint8x8_t, __builtin_riscv_pabd_i8x8)
__packed_binary_builtin_cast(pabd_i16x4, int16x4_t, uint16x4_t, __builtin_riscv_pabd_i16x4)
__packed_binary_builtin_cast(pabdu_u8x8, uint8x8_t, uint8x8_t, __builtin_riscv_pabdu_u8x8)
__packed_binary_builtin_cast(pabdu_u16x4, uint16x4_t, uint16x4_t, __builtin_riscv_pabdu_u16x4)

/* Packed Reduction Sum (32-bit) */
__packed_reduction(predsum_i8x4_i32, int32_t, int8x4_t, __builtin_riscv_predsum_i8x4_i32)
__packed_reduction(predsumu_u8x4_u32, uint32_t, uint8x4_t, __builtin_riscv_predsumu_u8x4_u32)
__packed_reduction(predsum_i16x2_i32, int32_t, int16x2_t, __builtin_riscv_predsum_i16x2_i32)
__packed_reduction(predsumu_u16x2_u32, uint32_t, uint16x2_t, __builtin_riscv_predsumu_u16x2_u32)

/* Packed Reduction Sum (64-bit) */
__packed_reduction(predsum_i8x8_i32, int32_t, int8x8_t, __builtin_riscv_predsum_i8x8_i32)
__packed_reduction(predsumu_u8x8_u32, uint32_t, uint8x8_t, __builtin_riscv_predsumu_u8x8_u32)
__packed_reduction(predsum_i16x4_i32, int32_t, int16x4_t, __builtin_riscv_predsum_i16x4_i32)
__packed_reduction(predsumu_u16x4_u32, uint32_t, uint16x4_t, __builtin_riscv_predsumu_u16x4_u32)
__packed_reduction(predsum_i8x8_i64, int64_t, int8x8_t, __builtin_riscv_predsum_i8x8_i64)
__packed_reduction(predsumu_u8x8_u64, uint64_t, uint8x8_t, __builtin_riscv_predsumu_u8x8_u64)
__packed_reduction(predsum_i16x4_i64, int64_t, int16x4_t, __builtin_riscv_predsum_i16x4_i64)
__packed_reduction(predsumu_u16x4_u64, uint64_t, uint16x4_t, __builtin_riscv_predsumu_u16x4_u64)
__packed_reduction(predsum_i32x2_i64, int64_t, int32x2_t, __builtin_riscv_predsum_i32x2_i64)
__packed_reduction(predsumu_u32x2_u64, uint64_t, uint32x2_t, __builtin_riscv_predsumu_u32x2_u64)

/* Packed Merge (32-bit) */
__packed_merge_builtin(pmerge_u8x4, uint8x4_t, uint8x4_t, __builtin_riscv_pmerge_u8x4)
__packed_merge_builtin(pmerge_i8x4, int8x4_t, uint8x4_t, __builtin_riscv_pmerge_i8x4)
__packed_merge_builtin(pmerge_u16x2, uint16x2_t, uint16x2_t, __builtin_riscv_pmerge_u16x2)
__packed_merge_builtin(pmerge_i16x2, int16x2_t, uint16x2_t, __builtin_riscv_pmerge_i16x2)

/* Packed Merge (64-bit) */
__packed_merge_builtin(pmerge_u8x8, uint8x8_t, uint8x8_t, __builtin_riscv_pmerge_u8x8)
__packed_merge_builtin(pmerge_i8x8, int8x8_t, uint8x8_t, __builtin_riscv_pmerge_i8x8)
__packed_merge_builtin(pmerge_u16x4, uint16x4_t, uint16x4_t, __builtin_riscv_pmerge_u16x4)
__packed_merge_builtin(pmerge_i16x4, int16x4_t, uint16x4_t, __builtin_riscv_pmerge_i16x4)
__packed_merge_builtin(pmerge_u32x2, uint32x2_t, uint32x2_t, __builtin_riscv_pmerge_u32x2)
__packed_merge_builtin(pmerge_i32x2, int32x2_t, uint32x2_t, __builtin_riscv_pmerge_i32x2)

/* Packed Absolute Difference Sum (32-bit) */
__packed_abdsum(pabdsumu_u8x4_u32, uint32_t, uint8x4_t, __builtin_riscv_pabdsumu_u8x4_u32)
__packed_abdsum_acc(pabdsumau_u8x4_u32, uint32_t, uint8x4_t, __builtin_riscv_pabdsumau_u8x4_u32)

/* Packed Absolute Difference Sum (64-bit) */
__packed_abdsum(pabdsumu_u8x8_u32, uint32_t, uint8x8_t, __builtin_riscv_pabdsumu_u8x8_u32)
__packed_abdsum(pabdsumu_u8x8_u64, uint64_t, uint8x8_t, __builtin_riscv_pabdsumu_u8x8_u64)
__packed_abdsum_acc(pabdsumau_u8x8_u32, uint32_t, uint8x8_t, __builtin_riscv_pabdsumau_u8x8_u32)
__packed_abdsum_acc(pabdsumau_u8x8_u64, uint64_t, uint8x8_t, __builtin_riscv_pabdsumau_u8x8_u64)

/* Packed Saturating Absolute Value (32-bit) */
__packed_psabs(psabs_i8x4, int8x4_t, __builtin_riscv_psabs_i8x4)
__packed_psabs(psabs_i16x2, int16x2_t, __builtin_riscv_psabs_i16x2)

/* Packed Saturating Absolute Value (64-bit) */
__packed_psabs(psabs_i8x8, int8x8_t, __builtin_riscv_psabs_i8x8)
__packed_psabs(psabs_i16x4, int16x4_t, __builtin_riscv_psabs_i16x4)

// clang-format on

#undef __packed_splat2
#undef __packed_splat4
#undef __packed_splat8
#undef __packed_splat
#undef __packed_shift
#undef __packed_shift8
#undef __packed_shift16
#undef __packed_shift32
#undef __packed_scalar_binary_op
#undef __packed_binary_op
#undef __packed_unary_op
#undef __packed_binary_builtin
#undef __packed_sh1add
#undef __packed_sh1sadd
#undef __packed_cmp
#undef __packed_pabs
#undef __packed_binary_builtin_cast
#undef __packed_reduction
#undef __packed_merge_builtin
#undef __packed_psabs
#undef __packed_reverse2
#undef __packed_reverse4
#undef __packed_reverse8
#undef __packed_zip2
#undef __packed_zip4
#undef __packed_unzipe2
#undef __packed_unzipe4
#undef __packed_unzipo2
#undef __packed_unzipo4
#undef __packed_abdsum
#undef __packed_abdsum_acc
#undef __DEFAULT_FN_ATTRS

#if defined(__cplusplus)
}
#endif

#endif /* __RISCV_PACKED_SIMD_H */
