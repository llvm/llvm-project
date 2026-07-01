/*===---- riscv_corev_simd.h - CORE-V SIMD intrinsics ----------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 *
 * This header provides C intrinsics for the CORE-V XCVsimd ISA extension.
 * Include this header when compiling with -march=..._xcvsimd.
 *
 * All operations work on packed sub-word elements within a 32-bit GPR:
 *   - .h variants operate on two packed int16 / uint16 (halfwords)
 *   - .b variants operate on four packed int8  / uint8  (bytes)
 *
 * Element-wise ops (add/sub, min/max, and/or/xor, abs) can also be written
 * directly on GCC-style packed vector types; these wrappers match the
 * CORE-V GCC builtin spellings.
 *
 * Spec: https://docs.openhwgroup.org/projects/cv32e40p-user-manual/en/latest/
 *       instruction_set_extensions.html#simd
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __RISCV_COREV_SIMD_H
#define __RISCV_COREV_SIMD_H

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__riscv_xcvsimd)

#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __artificial__))

/* ADD / SUB
 * ---------------------------------------------------------------------- */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_add_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_add_h(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_add_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_add_b(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sub_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_sub_h(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sub_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_sub_b(a0, a1);
}

/* MIN / MINU / MAX / MAXU
 * ---------------------------------------------------------------------- */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_min_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_min_h(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_min_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_min_b(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_minu_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_minu_h(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_minu_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_minu_b(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_max_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_max_h(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_max_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_max_b(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_maxu_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_maxu_h(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_maxu_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_maxu_b(a0, a1);
}

/* AND / OR / XOR
 * ---------------------------------------------------------------------- */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_and_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_and_h(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_and_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_and_b(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_or_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_or_h(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_or_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_or_b(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_xor_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_xor_h(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_xor_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_xor_b(a0, a1);
}

/* ABS
 * ---------------------------------------------------------------------- */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_abs_h(uint32_t a0) {
  return __builtin_riscv_cv_simd_abs_h(a0);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_abs_b(uint32_t a0) {
  return __builtin_riscv_cv_simd_abs_b(a0);
}

/* DOT PRODUCTS
 * ---------------------------------------------------------------------- */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotup_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_dotup_h(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotup_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_dotup_b(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotup_sc_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_dotup_sc_h(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotup_sc_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_dotup_sc_b(a0, a1);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotusp_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_dotusp_h(a0, a1);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotusp_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_dotusp_b(a0, a1);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotusp_sc_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_dotusp_sc_h(a0, a1);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotusp_sc_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_dotusp_sc_b(a0, a1);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotsp_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_dotsp_h(a0, a1);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotsp_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_dotsp_b(a0, a1);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotsp_sc_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_dotsp_sc_h(a0, a1);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_dotsp_sc_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_dotsp_sc_b(a0, a1);
}

/* SDOT (accumulating)
 * ---------------------------------------------------------------------- */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotup_h(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __builtin_riscv_cv_simd_sdotup_h(a0, a1, a2);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotup_b(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __builtin_riscv_cv_simd_sdotup_b(a0, a1, a2);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotup_sc_h(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __builtin_riscv_cv_simd_sdotup_sc_h(a0, a1, a2);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotup_sc_b(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __builtin_riscv_cv_simd_sdotup_sc_b(a0, a1, a2);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotusp_h(uint32_t a0, uint32_t a1, int32_t a2) {
  return __builtin_riscv_cv_simd_sdotusp_h(a0, a1, a2);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotusp_b(uint32_t a0, uint32_t a1, int32_t a2) {
  return __builtin_riscv_cv_simd_sdotusp_b(a0, a1, a2);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotusp_sc_h(uint32_t a0, uint32_t a1, int32_t a2) {
  return __builtin_riscv_cv_simd_sdotusp_sc_h(a0, a1, a2);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotusp_sc_b(uint32_t a0, uint32_t a1, int32_t a2) {
  return __builtin_riscv_cv_simd_sdotusp_sc_b(a0, a1, a2);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotsp_h(uint32_t a0, uint32_t a1, int32_t a2) {
  return __builtin_riscv_cv_simd_sdotsp_h(a0, a1, a2);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotsp_b(uint32_t a0, uint32_t a1, int32_t a2) {
  return __builtin_riscv_cv_simd_sdotsp_b(a0, a1, a2);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotsp_sc_h(uint32_t a0, uint32_t a1, int32_t a2) {
  return __builtin_riscv_cv_simd_sdotsp_sc_h(a0, a1, a2);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_sdotsp_sc_b(uint32_t a0, uint32_t a1, int32_t a2) {
  return __builtin_riscv_cv_simd_sdotsp_sc_b(a0, a1, a2);
}

/* EXTRACT / INSERT
 * ---------------------------------------------------------------------- */

#define __riscv_cv_simd_extract_h(__a0, __a1)                                  \
  ((int32_t)__builtin_riscv_cv_simd_extract_h((uint32_t)(__a0),                \
                                              (uint32_t)(__a1)))

#define __riscv_cv_simd_extract_b(__a0, __a1)                                  \
  ((int32_t)__builtin_riscv_cv_simd_extract_b((uint32_t)(__a0),                \
                                              (uint32_t)(__a1)))

#define __riscv_cv_simd_extractu_h(__a0, __a1)                                 \
  ((uint32_t)__builtin_riscv_cv_simd_extractu_h((uint32_t)(__a0),              \
                                                (uint32_t)(__a1)))

#define __riscv_cv_simd_extractu_b(__a0, __a1)                                 \
  ((uint32_t)__builtin_riscv_cv_simd_extractu_b((uint32_t)(__a0),              \
                                                (uint32_t)(__a1)))

#define __riscv_cv_simd_insert_h(__a0, __a1, __a2)                             \
  ((uint32_t)__builtin_riscv_cv_simd_insert_h(                                 \
      (uint32_t)(__a0), (uint32_t)(__a1), (uint32_t)(__a2)))

#define __riscv_cv_simd_insert_b(__a0, __a1, __a2)                             \
  ((uint32_t)__builtin_riscv_cv_simd_insert_b(                                 \
      (uint32_t)(__a0), (uint32_t)(__a1), (uint32_t)(__a2)))

/* SHUFFLE
 * ---------------------------------------------------------------------- */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_shuffle_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_shuffle_h(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_shuffle_b(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_shuffle_b(a0, a1);
}

#define __riscv_cv_simd_shuffle_sci_h(__a0, __a1)                              \
  ((uint32_t)__builtin_riscv_cv_simd_shuffle_sci_h((uint32_t)(__a0),           \
                                                   (uint32_t)(__a1)))

#define __riscv_cv_simd_shuffle_sci_b(__a0, __a1)                              \
  ((uint32_t)__builtin_riscv_cv_simd_shuffle_sci_b((uint32_t)(__a0),           \
                                                   (uint32_t)(__a1)))

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_shuffle2_h(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __builtin_riscv_cv_simd_shuffle2_h(a0, a1, a2);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_shuffle2_b(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __builtin_riscv_cv_simd_shuffle2_b(a0, a1, a2);
}

/* PACK
 * ---------------------------------------------------------------------- */

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_packhi_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_packhi_h(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_packlo_h(uint32_t a0, uint32_t a1) {
  return __builtin_riscv_cv_simd_packlo_h(a0, a1);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_packhi_b(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __builtin_riscv_cv_simd_packhi_b(a0, a1, a2);
}

static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_simd_packlo_b(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __builtin_riscv_cv_simd_packlo_b(a0, a1, a2);
}

/* COMPLEX
 * ---------------------------------------------------------------------- */

#define __riscv_cv_simd_cplxmul_r(__a0, __a1, __a2, __a3)                      \
  ((uint32_t)__builtin_riscv_cv_simd_cplxmul_r(                                \
      (uint32_t)(__a0), (uint32_t)(__a1), (uint32_t)(__a2), (uint32_t)(__a3)))

#define __riscv_cv_simd_cplxmul_i(__a0, __a1, __a2, __a3)                      \
  ((uint32_t)__builtin_riscv_cv_simd_cplxmul_i(                                \
      (uint32_t)(__a0), (uint32_t)(__a1), (uint32_t)(__a2), (uint32_t)(__a3)))

static __inline__ uint32_t
    __DEFAULT_FN_ATTRS __riscv_cv_simd_cplxconj(uint32_t a0) {
  return __builtin_riscv_cv_simd_cplxconj(a0);
}

#define __riscv_cv_simd_subrotmj(__a0, __a1, __a2)                             \
  ((uint32_t)__builtin_riscv_cv_simd_subrotmj(                                 \
      (uint32_t)(__a0), (uint32_t)(__a1), (uint32_t)(__a2)))

#endif /* __riscv_xcvsimd */

#if defined(__cplusplus)
}
#endif

#endif /* __RISCV_COREV_SIMD_H */
