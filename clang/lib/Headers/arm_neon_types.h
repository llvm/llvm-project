/*===---- arm_neon_types.h - ARM NEON TYPES --------------------------------===
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __ARM_NEON_TYPES_H
#define __ARM_NEON_TYPES_H
#ifdef __cplusplus
extern "C" {
#endif
#ifndef __ARM_NEON_H
typedef __attribute__((vector_size(16))) int8_t int8x16_t;
typedef __attribute__((vector_size(16))) int16_t int16x8_t;
typedef __attribute__((vector_size(16))) int32_t int32x4_t;
typedef __attribute__((vector_size(16))) int64_t int64x2_t;
typedef __attribute__((vector_size(16))) uint8_t uint8x16_t;
typedef __attribute__((vector_size(16))) uint16_t uint16x8_t;
typedef __attribute__((vector_size(16))) uint32_t uint32x4_t;
typedef __attribute__((vector_size(16))) uint64_t uint64x2_t;
typedef __attribute__((vector_size(16))) float16_t float16x8_t;
typedef __attribute__((vector_size(16))) float32_t float32x4_t;
typedef __attribute__((vector_size(16))) float64_t float64x2_t;
#else
typedef __attribute__((neon_vector_type(8))) int8_t int8x8_t;
typedef __attribute__((neon_vector_type(4))) int16_t int16x4_t;
typedef __attribute__((neon_vector_type(2))) int32_t int32x2_t;
typedef __attribute__((neon_vector_type(1))) int64_t int64x1_t;
typedef __attribute__((neon_vector_type(8))) uint8_t uint8x8_t;
typedef __attribute__((neon_vector_type(4))) uint16_t uint16x4_t;
typedef __attribute__((neon_vector_type(2))) uint32_t uint32x2_t;
typedef __attribute__((neon_vector_type(1))) uint64_t uint64x1_t;
typedef __attribute__((neon_vector_type(4))) float16_t float16x4_t;
typedef __attribute__((neon_vector_type(2))) float32_t float32x2_t;
#ifdef __aarch64__
typedef __attribute__((neon_vector_type(1))) float64_t float64x1_t;
#endif
typedef __attribute__((neon_vector_type(16))) int8_t int8x16_t;
typedef __attribute__((neon_vector_type(8))) int16_t int16x8_t;
typedef __attribute__((neon_vector_type(4))) int32_t int32x4_t;
typedef __attribute__((neon_vector_type(2))) int64_t int64x2_t;
typedef __attribute__((neon_vector_type(16))) uint8_t uint8x16_t;
typedef __attribute__((neon_vector_type(8))) uint16_t uint16x8_t;
typedef __attribute__((neon_vector_type(4))) uint32_t uint32x4_t;
typedef __attribute__((neon_vector_type(2))) uint64_t uint64x2_t;
typedef __attribute__((neon_vector_type(8))) float16_t float16x8_t;
typedef __attribute__((neon_vector_type(4))) float32_t float32x4_t;
#ifdef __aarch64__
typedef __attribute__((neon_vector_type(2))) float64_t float64x2_t;
#endif
typedef __attribute__((neon_vector_type(4))) bfloat16_t bfloat16x4_t;
typedef __attribute__((neon_vector_type(8))) bfloat16_t bfloat16x8_t;
#endif
#ifdef __cplusplus
} // extern "C"
#endif
#endif //__ARM_NEON_TYPES_H
