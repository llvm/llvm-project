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

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__))

#if defined(__riscv_xandesperf)

#if __riscv_xlen == 32

static __inline__ int32_t __DEFAULT_FN_ATTRS __riscv_nds_ffb_32(uint32_t __a,
                                                                uint32_t __b) {
  return __builtin_riscv_nds_ffb_32(__a, __b);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS __riscv_nds_ffzmism_32(uint32_t __a,
                                                                    uint32_t __b) {
  return __builtin_riscv_nds_ffzmism_32(__a, __b);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS __riscv_nds_ffmism_32(uint32_t __a,
                                                                   uint32_t __b) {
  return __builtin_riscv_nds_ffmism_32(__a, __b);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS __riscv_nds_flmism_32(uint32_t __a,
                                                                   uint32_t __b) {
  return __builtin_riscv_nds_flmism_32(__a, __b);
}

#endif

#if __riscv_xlen == 64

static __inline__ int64_t __DEFAULT_FN_ATTRS __riscv_nds_ffb_64(uint64_t __a,
                                                                uint64_t __b) {
  return __builtin_riscv_nds_ffb_64(__a, __b);
}

static __inline__ int64_t __DEFAULT_FN_ATTRS __riscv_nds_ffzmism_64(uint64_t __a,
                                                                    uint64_t __b) {
  return __builtin_riscv_nds_ffzmism_64(__a, __b);
}

static __inline__ int64_t __DEFAULT_FN_ATTRS __riscv_nds_ffmism_64(uint64_t __a,
                                                                   uint64_t __b) {
  return __builtin_riscv_nds_ffmism_64(__a, __b);
}

static __inline__ int64_t __DEFAULT_FN_ATTRS __riscv_nds_flmism_64(uint64_t __a,
                                                                   uint64_t __b) {
  return __builtin_riscv_nds_flmism_64(__a, __b);
}

#endif

#endif // defined(__riscv_xandesperf)

#if defined(__riscv_xandesbfhcvt)

static __inline__ float __DEFAULT_FN_ATTRS __riscv_nds_fcvt_s_bf16(__bf16 bf) {
  return __builtin_riscv_nds_fcvt_s_bf16(bf);
}

static __inline__ __bf16 __DEFAULT_FN_ATTRS __riscv_nds_fcvt_bf16_s(float sf) {
  return __builtin_riscv_nds_fcvt_bf16_s(sf);
}

#endif // defined(__riscv_xandesbfhcvt)

#if defined(__cplusplus)
}
#endif

#endif // define __RISCV_NDS_H
